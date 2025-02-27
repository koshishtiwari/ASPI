"""Date parsing and validation utilities."""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Union, Optional

def safe_convert_to_float(value: Any) -> float:
    """Safely convert a value to float, handling various formats.
    
    Args:
        value: The value to convert
        
    Returns:
        The float value or 0.0 if conversion fails
    """
    if pd.isna(value):
        return 0.0
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        # Try to remove any non-numeric characters except decimal point and negative sign
        try:
            # Check if it might be a date
            if '/' in value or '-' in value or ',' in value:
                if is_date(value):
                    return 0.0
            
            # Remove non-numeric characters except decimal point and negative sign
            numeric_str = ''.join(c for c in value if c.isdigit() or c in '.-')
            return float(numeric_str) if numeric_str else 0.0
        except:
            return 0.0
    
    return 0.0

def is_date(value: Any) -> bool:
    """Check if a value is a date string.
    
    Args:
        value: The value to check
        
    Returns:
        True if the value is a date, False otherwise
    """
    if not isinstance(value, str):
        return False
    
    date_formats = [
        '%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y',
        '%b %d, %Y', '%B %d, %Y', '%d %b, %Y', '%d %B, %Y',
        '%m-%d-%Y', '%d-%m-%Y', '%Y%m%d'
    ]
    
    for fmt in date_formats:
        try:
            datetime.strptime(value, fmt)
            return True
        except ValueError:
            continue
    
    return False

def clean_dataframe(df: pd.DataFrame, numeric_cols: Optional[list] = None) -> pd.DataFrame:
    """Clean a DataFrame by converting all numeric columns to float safely.
    
    Args:
        df: The DataFrame to clean
        numeric_cols: Optional list of columns to convert to numeric, if None all columns are checked
        
    Returns:
        Cleaned DataFrame
    """
    # Create a copy
    clean_df = df.copy()
    
    # If no specific columns provided, try to convert all columns
    if numeric_cols is None:
        numeric_cols = []
        for col in df.columns:
            # Skip date columns
            if df[col].dtype == 'datetime64[ns]' or (df[col].dtype == 'object' and df[col].apply(is_date).all()):
                continue
            numeric_cols.append(col)
    
    # Convert each numeric column
    for col in numeric_cols:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].apply(safe_convert_to_float)
    
    return clean_df

def parse_date_column(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Parse a date column to datetime format.
    
    Args:
        df: DataFrame containing the date column
        date_col: Name of the column containing dates
        
    Returns:
        DataFrame with parsed date column
    """
    # Create a copy
    parsed_df = df.copy()
    
    # Check if column exists
    if date_col not in parsed_df.columns:
        return parsed_df
    
    # Try common date formats
    try:
        parsed_df[date_col] = pd.to_datetime(parsed_df[date_col], errors='coerce')
    except Exception as e:
        print(f"Error parsing dates: {e}")
    
    # Drop rows with failed date parsing
    parsed_df = parsed_df.dropna(subset=[date_col])
    
    return parsed_df