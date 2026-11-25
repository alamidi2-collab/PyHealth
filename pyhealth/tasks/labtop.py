"""LabTOP's numeric eICU and HiRID tasks, operating on PyHealth Patient events."""

from abc import ABC, abstractmethod
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import time, timedelta
from functools import lru_cache
from itertools import groupby
import math

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from pyhealth.data import Patient


# PyHealth names tables in lowercase. Preserve LabTOP's text in the sequence.
_TABLES = {
    "lab": ("lab", "labresultoffset", "labname"),
    "medication": ("medication", "drugstartoffset", "drugname"),
    "treatment": ("treatment", "treatmentoffset", "treatmentstring"),
    "infusiondrug": ("infusionDrug", "infusionoffset", "drugname"),
    "intakeoutput": ("intakeOutput", "intakeoutputoffset", "celllabel"),
}
_WEEKDAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
END = "|endofevent|"


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


@dataclass
class _Event:
    offset: float
    time_tokens: tuple
    table: str
    item: str
    value: str
    unit: str


class _LabTOPProcessorBase(ABC):
    """Shared training sequences and prediction prompts for LabTOP."""

    weekdays = ()
    lab_table: str

    def __init__(self, tokenizer, max_seq_len=4096, max_days=507):
        if tokenizer.pad_token_id != 0 or max_seq_len < 2 or max_days < 1:
            raise ValueError("Require padding ID zero, positive max_days and length >= 2")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_days = max_days
        self.selected_items = None
        special = ["[unused1]"] + [f"[NUM_{n}]" for n in range(1, 6)]
        days = [f"[DAY_{n}]" for n in range(1, max_days + 1)]
        special += days
        special += [f"[{day}]" for day in self.weekdays]
        self.time_prefix_length = 4 if self.weekdays else 3
        special += [f"[{h:02}h]" for h in range(24)]
        special += [f"[{m:02}m]" for m in range(0, 60, 10)]
        tokenizer.add_special_tokens({"additional_special_tokens": special + [END]})
        self.day_ids = set(tokenizer.convert_tokens_to_ids(days))
        self.encode = lru_cache(maxsize=32768)(
            lambda field: tokenizer.encode(field, add_special_tokens=False)
        )

    @abstractmethod
    def _stays(self, patient):
        """Yield (stay ID, chronological events) for eligible admissions."""

    def fit(self, patients):
        """Select items covering 90% of each table's training events, including ties."""
        counts = defaultdict(Counter)
        for patient in patients:
            for _, events in self._stays(patient):
                for event in events:
                    counts[event.table][event.item] += 1
        selected = set()
        for table, frequencies in counts.items():
            total = sum(frequencies.values())
            cumulative = 0
            for _, cutoff in frequencies.most_common():
                cumulative += cutoff
                if cumulative >= 0.9 * total:
                    break
            selected.update((table, item) for item, n in frequencies.items() if n >= cutoff)
        self.selected_items = selected

    def _selected_stays(self, patient):
        for stay, events in self._stays(patient):
            if self.selected_items is not None:
                events = [e for e in events if (e.table, e.item) in self.selected_items]
            yield stay, events

    def _tokens(self, event):
        header = [*event.time_tokens, event.table]
        tokens = [i for field in header for i in self.encode(field)]
        item_start = len(tokens)
        tokens.extend(self.encode(event.item))
        value_start = len(tokens)
        for field in (event.value, event.unit, END):
            if field:
                tokens.extend(self.encode(field))
        types = [0] * item_start + [int(event.table == self.lab_table)] * (len(tokens) - item_start)
        return tokens, types, value_start

    def preprocess(self, patient: Patient, lab_only=False):
        """Return one training/validation row per stay with a supervised token."""
        rows = []
        for stay, events in self._selected_stays(patient):
            tokens, types = [], []
            for event in events:
                ids, mask, _ = self._tokens(event)
                if len(tokens) + len(ids) > self.max_seq_len:
                    break
                tokens.extend(ids)
                types.extend(mask)
            if tokens and (not lab_only or any(types)):
                rows.append(dict(patient_id=patient.patient_id, icu_id=stay,
                                 tokens=tokens, types=types))
        return rows

    def collate_train(self, rows):
        """Shuffle complete events within each time bin, carrying their masks."""
        shuffled = []
        for row in rows:
            tokens, types = row["tokens"], row["types"]
            starts = [i for i, token in enumerate(tokens) if token in self.day_ids]
            spans = zip(starts, starts[1:] + [len(tokens)])
            order = []
            for _, group in groupby(spans, key=lambda span: tokens[span[0]:span[0] + self.time_prefix_length]):
                group = list(group)
                order.extend(group[i] for i in torch.randperm(len(group)).tolist())
            shuffled.append({"tokens": [x for a, b in order for x in tokens[a:b]],
                             "types": [x for a, b in order for x in types[a:b]]})
        return collate_labtop(shuffled)

    def make_inference_dataset(self, patient: Patient):
        """Yield held-out lab targets with whole, strictly earlier time bins."""
        for stay, events in self._selected_stays(patient):
            history = deque()
            length = 0
            previous = {}
            for _, group in groupby(events, key=lambda event: event.time_tokens):
                current = []
                for event in group:
                    ids, types, start = self._tokens(event)
                    value = _number(event.value)
                    if event.table == self.lab_table and value is not None:
                        prev, _, mean = previous.get(event.item, ([], 0, math.nan))
                        # Reserve a fixed generation budget independent of the answer.
                        budget = self.max_seq_len - start - min(50, self.max_seq_len // 4)
                        context = []
                        available = budget
                        for past in reversed(history):
                            if len(past) > available:
                                break
                            context.append(past)
                            available -= len(past)
                        if context:
                            yield dict(patient_id=patient.patient_id, icu_id=stay,
                                       prompt_tokens=[i for past in reversed(context) for i in past],
                                       lab_tokens=ids, lab_types=types, value_start=start,
                                       lab_value=value, item=event.item, unit=event.unit,
                                       previous_value_tokens=prev, mean_previous_values=mean)
                    current.append((event, ids, value))
                for event, ids, value in current:
                    history.append(ids)
                    length += len(ids)
                    while length > self.max_seq_len:
                        length -= len(history.popleft())
                    if event.table == self.lab_table and value is not None:
                        _, count, mean = previous.get(event.item, ([], 0, 0.0))
                        previous[event.item] = (
                            ids[:-1], count + 1, round((mean * count + value) / (count + 1), 2)
                        )


class LabTOPProcessor(_LabTOPProcessorBase):
    """Convert eICU Patient objects to LabTOP sequences and lab prompts."""

    lab_table = "lab"

    def _stays(self, patient):
        bounds = {}
        for event in patient.get_events(event_type="patient"):
            attrs = event.attr_dict
            try:
                clock = time.fromisoformat(str(attrs.get("unitadmittime24")))
            except ValueError:
                continue
            end = _number(attrs.get("unitdischargeoffset"))
            if end is not None and end >= 360 and attrs.get("age") is not None:
                bounds[attrs["patientunitstayid"]] = (clock.hour * 60 + clock.minute, end)

        stays = defaultdict(list)
        seen = set()
        for source, (table, offset_key, item_key) in _TABLES.items():
            for event in patient.get_events(event_type=source):
                attrs = event.attr_dict
                stay = attrs.get("patientunitstayid")
                offset = _number(attrs.get(offset_key))
                item = str(attrs.get(item_key) or "").strip().lower()
                if stay not in bounds or offset is None or item in ("", "___", "_ _ _"):
                    continue
                start, end = bounds[stay]
                key = (stay, table, offset, item)
                if not 0 <= offset <= end or key in seen:
                    continue
                seen.add(key)
                day, minute = divmod(int(start + offset), 1440)
                hour, minute = divmod(minute, 60)
                stamp = (f"[DAY_{min(day + 1, self.max_days)}]",
                         f"[{hour:02}h]", f"[{minute // 10 * 10:02}m]")
                value = attrs.get("labresult")
                unit = attrs.get("labmeasurenamesystem") or attrs.get("labmeasurenameinterface")
                stays[stay].append(_Event(offset, stamp, table, item,
                                          str(value) if value is not None else "", unit or ""))
        for stay, events in stays.items():
            yield stay, sorted(events, key=lambda event: event.offset)


class LabTOPHiRIDProcessor(_LabTOPProcessorBase):
    """Convert HiRID Patient objects using the raw-stage variable reference.

    HiRID identifies one ICU admission per patientid. As in LabTOP, all
    numeric observations are eligible prediction targets, including vitals.
    Medication names and dose units provide historical context. Length of stay is measured from
    admission to the latest timestamp in either raw event table.
    """

    lab_table = "observation_tables"
    weekdays = _WEEKDAYS

    def __init__(self, tokenizer, variable_reference_path, max_seq_len=4096, max_days=250):
        super().__init__(tokenizer, max_seq_len, max_days)
        columns = ["Source Table", "ID", "Variable Name", "Unit"]
        try:
            reference = pd.read_csv(variable_reference_path, usecols=columns, encoding="utf-8-sig")
        except UnicodeDecodeError:
            reference = pd.read_csv(variable_reference_path, usecols=columns, encoding="latin-1")
        self.variables = {}
        for source, code, name, unit in reference[columns].itertuples(index=False, name=None):
            if pd.notna(code) and pd.notna(name):
                self.variables[str(source).strip().lower(), int(code)] = (
                    str(name).strip().lower(), "" if pd.isna(unit) else str(unit).strip()
                )

    def _stays(self, patient):
        admissions = patient.get_events("general_table")
        if not admissions or admissions[0].timestamp is None:
            return
        admission = admissions[0].timestamp
        last = admission
        events = []
        seen = set()
        for table, source, code_key in (
            (self.lab_table, "observation", "variableid"),
            ("pharma_records", "pharma", "pharmaid"),
        ):
            for row in patient.get_events(table):
                when = row.timestamp
                if when is None or when < admission:
                    continue
                last = max(last, when)
                attrs = row.attr_dict
                variable = self.variables.get((source, attrs[code_key]))
                if variable is None:
                    continue
                name, unit = variable
                # Distinct raw codes can map to the same descriptive name.
                key = (when, table, name)
                if key in seen:
                    continue
                seen.add(key)
                if source == "pharma" and attrs.get("doseunit") is not None:
                    unit = str(attrs["doseunit"]).strip() or unit
                # LabTOP serializes observation values; pharma contributes its name and unit.
                value = attrs.get("value") if table == self.lab_table else None
                day = min((when.date() - admission.date()).days + 1, self.max_days)
                stamp = (f"[DAY_{day}]", f"[{self.weekdays[when.weekday()]}]",
                         f"[{when.hour:02}h]", f"[{when.minute // 10 * 10:02}m]")
                events.append(_Event(
                    (when - admission).total_seconds() / 60, stamp, table, name,
                    str(value) if _number(value) is not None else "", unit
                ))
        if events and last - admission >= timedelta(hours=6):
            yield patient.patient_id, sorted(events, key=lambda event: event.offset)


def collate_labtop(rows):
    """Right-pad token IDs, lab supervision masks and positions to [B, L]."""
    return tuple(pad_sequence(values, batch_first=True) for values in (
        [torch.tensor(row["tokens"], dtype=torch.long) for row in rows],
        [torch.tensor(row["types"], dtype=torch.long) for row in rows],
        [torch.arange(len(row["tokens"])) for row in rows],
    ))


def collate_labtop_prompts(rows):
    """Build right-padded prediction prefixes and keep targets out of model inputs."""
    prompts = [torch.tensor(row["prompt_tokens"] + row["lab_tokens"][:row["value_start"]])
               for row in rows]
    previous = [torch.tensor(row["previous_value_tokens"], dtype=torch.long) for row in rows]
    labels = [torch.tensor(row["lab_tokens"][row["value_start"]:]) for row in rows]
    return (pad_sequence(prompts, batch_first=True),
            pad_sequence(labels, batch_first=True),
            pad_sequence(previous, batch_first=True),
            torch.tensor([row["mean_previous_values"] for row in rows], dtype=torch.float64))
