import os
import json
import glob
import pandas as pd
from datetime import datetime

def preprocess_flight_data():
    """
    Aggregates and preprocesses all raw flight data JSON files.
    Implements data merging and deduplication to maintain a single, 
    continuous historical dataset for model training.
    """
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    json_files = glob.glob(os.path.join(raw_dir, "*.json"))
    if not json_files:
        print("[ERROR] No raw data found. Please execute data ingestion first.")
        return

    print(f"Discovered {len(json_files)} raw data file(s). Initiating aggregation and preprocessing...")

    processed_records = []

    # Iterate through all available raw data files to merge historical data
    for file_path in json_files:
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                print(f"[WARNING] Skipping unreadable JSON file: {file_path}")
                continue

        if 'data' not in data:
            continue

        flights = data['data']
        for flight in flights:
            try:
                flight_status = flight.get('flight_status')
                scheduled_arr = flight.get('arrival', {}).get('scheduled')
                actual_arr = flight.get('arrival', {}).get('actual')

                # Filter out cancelled flights or records missing crucial time data
                if flight_status not in ['landed', 'active'] or not scheduled_arr or not actual_arr:
                    continue

                # Parse ISO 8601 datetime strings
                sch_time = datetime.fromisoformat(scheduled_arr.replace('Z', '+00:00'))
                act_time = datetime.fromisoformat(actual_arr.replace('Z', '+00:00'))
                
                # Calculate delay in minutes
                delay_minutes = (act_time - sch_time).total_seconds() / 60.0
                
                # Create binary target variable (1 if delayed > 15 mins, else 0)
                is_delayed = 1 if delay_minutes > 15 else 0

                processed_records.append({
                    'flight_date': flight.get('flight_date'),
                    'airline': flight.get('airline', {}).get('name'),
                    'flight_number': flight.get('flight', {}).get('number'),
                    'departure_iata': flight.get('departure', {}).get('iata'),
                    'arrival_iata': flight.get('arrival', {}).get('iata'),
                    'scheduled_arrival': scheduled_arr,
                    'actual_arrival': actual_arr,
                    'delay_minutes': round(delay_minutes, 2),
                    'is_delayed': is_delayed
                })
            except Exception:
                continue

    if not processed_records:
        print("[WARNING] No valid flight records found across all files.")
        return

    # Convert aggregated records into a Pandas DataFrame
    df = pd.DataFrame(processed_records)
    
    # Deduplicate records to prevent data leakage and redundancy
    initial_count = len(df)
    df.drop_duplicates(
        subset=['flight_date', 'airline', 'flight_number', 'scheduled_arrival'], 
        keep='last', 
        inplace=True
    )
    dedup_count = initial_count - len(df)
    
    if dedup_count > 0:
        print(f"Dropped {dedup_count} duplicated records during merging.")

    # Save to a static filename to ensure CI/CD and DVC track a single evolving file
    output_file = os.path.join(processed_dir, "processed_flights.csv")
    df.to_csv(output_file, index=False)
    
    print(f"[SUCCESS] Processed and merged {len(df)} unique records. Saved to {output_file}")

if __name__ == "__main__":
    preprocess_flight_data()