import os
import json
import glob
import pandas as pd
from datetime import datetime

def preprocess_flight_data():
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    json_files = glob.glob(os.path.join(raw_dir, "*.json"))
    if not json_files:
        print("Error: No raw data found. Please run ingest_data.py first.")
        return

    latest_file = max(json_files, key=os.path.getctime)
    print(f"Processing latest raw data file: {latest_file}")

    with open(latest_file, 'r') as file:
        data = json.load(file)

    if 'data' not in data:
        print("Error: Invalid JSON format. Missing 'data' array.")
        return

    flights = data['data']
    processed_records = []

    for flight in flights:
        try:
            flight_status = flight.get('flight_status')
            scheduled_arr = flight.get('arrival', {}).get('scheduled')
            actual_arr = flight.get('arrival', {}).get('actual')

            # Filter out cancelled flights or missing time data
            if flight_status not in ['landed', 'active'] or not scheduled_arr or not actual_arr:
                continue

            # Parse ISO 8601 datetime strings
            sch_time = datetime.fromisoformat(scheduled_arr.replace('Z', '+00:00'))
            act_time = datetime.fromisoformat(actual_arr.replace('Z', '+00:00'))
            
            # Calculate delay in minutes
            delay_minutes = (act_time - sch_time).total_seconds() / 60.0
            
            # Create binary target variable (1 if delayed > 30 mins, else 0)
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
        except Exception as e:
            continue

    if not processed_records:
        print("Warning: No valid flight records found to process.")
        return

    df = pd.DataFrame(processed_records)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(processed_dir, f"processed_flights_{timestamp}.csv")
    
    df.to_csv(output_file, index=False)
    print(f"Success: Processed {len(df)} records and saved to {output_file}")

if __name__ == "__main__":
    preprocess_flight_data()