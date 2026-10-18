"""Frozen source ontology; safe on a GPU host with older unrelated project code.

IDs and order copied from ml/export_human_dataset.py, human_observed_actions_v1.
This is the complete fixed ontology, not classes selected from evaluation labels.
"""

TAXONOMY_VERSION = "human_observed_actions_v1"
TAXONOMY_SOURCE = "ml/export_human_dataset.py"
TAXONOMY_SOURCE_SHA256 = "7465755c5d2e27bc12c6488d7b09bc4d5a8414a307731da632948f275ef4eecb"
ACTION_CODES = (
    "person_enters", "person_leaves", "leave_workstation", "return_to_workstation",
    "cash_received", "cash_transferred", "cash_placed", "cash_removed", "cash_visible",
    "cash_unpacking", "cash_bundling", "cash_counter_loading", "cash_counting",
    "cash_detector_check", "cash_stored", "document_transfer", "document_handling",
    "document_checked", "document_stored", "key_handling", "key_left",
    "stamp_handling", "stamp_stored", "safe_opening", "safe_closing", "safe_locking",
    "safe_open_visible", "computer_work", "phone_used", "phone_put_away",
    "drink_at_workstation", "object_handling", "other_observed_action",
)
ACTION_TO_ID = {action: index for index, action in enumerate(ACTION_CODES)}
