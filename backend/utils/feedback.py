"""
Feedback Utility
Handles saving user feedback and report data to CSV files
"""
import csv
import os
from pathlib import Path

from backend.core.config import Config
from backend.core.logger import setup_logger

logger = setup_logger(__name__)


def save_feedback_to_csv(
    message_id: str,
    response: str,
    feedback: str,
    csv_filename: str = None
):
    """
    Save user feedback to CSV file
    
    Args:
        message_id: Unique message identifier
        response: Model response
        feedback: User feedback
        csv_filename: Optional custom filename
    """
    if csv_filename is None:
        csv_filename = Config.FEEDBACK_PATH / "feedback_data.csv"
    else:
        csv_filename = Path(csv_filename)
    
    # Ensure directory exists
    csv_filename.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists
    file_exists = csv_filename.exists()
    
    # Write to CSV
    with open(csv_filename, "a", newline="") as csvfile:
        fieldnames = ["MessageID", "Response", "Feedback"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(
            {
                "MessageID": message_id,
                "Response": response,
                "Feedback": feedback,
            }
        )
    
    logger.info(f"Feedback saved to {csv_filename}")


def save_report_to_csv(
    message_id: str,
    response: str,
    csv_filename: str = None
):
    """
    Save report data to CSV file
    
    Args:
        message_id: Unique message identifier
        response: Model response
        csv_filename: Optional custom filename
    """
    if csv_filename is None:
        csv_filename = Config.REPORTS_PATH / "report_data.csv"
    else:
        csv_filename = Path(csv_filename)
    
    # Ensure directory exists
    csv_filename.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists
    file_exists = csv_filename.exists()
    
    # Write to CSV
    with open(csv_filename, "a", newline="") as csvfile:
        fieldnames = ["MessageID", "Response"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(
            {
                "MessageID": message_id,
                "Response": response,
            }
        )
    
    logger.info(f"Report data saved to {csv_filename}")

