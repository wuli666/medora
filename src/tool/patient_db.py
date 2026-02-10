"""Re-export patient DB functions for convenience."""

from src.utils.db import (
    get_follow_up_plans,
    get_or_create_patient,
    get_records,
    save_follow_up_plan,
    save_record,
)

__all__ = [
    "get_or_create_patient",
    "save_record",
    "get_records",
    "save_follow_up_plan",
    "get_follow_up_plans",
]
