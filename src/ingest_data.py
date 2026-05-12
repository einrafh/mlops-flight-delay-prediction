import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

def fetch_flight_data():
    api_key = os.getenv("AVIATIONSTACK_API_KEY")
    
    if not api_key or api_key == "your_api_key_here":
        raise ValueError("Missing API Key: Please set AVIATIONSTACK_API_KEY in your .env file.")

    url = "http://api.aviationstack.com/v1/flights"
    
    params = {
        'access_key': api_key,
        'dep_iata': 'SUB',
        'arr_iata': 'BDJ'
    }

    print("Fetching flight data from Aviationstack API...")
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        output_dir = os.path.join("data", "raw")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"flight_data_SUB_BDJ_{timestamp}.json")
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
            
        print(f"Success: Raw data saved to {filename}")
        
        if 'pagination' in data:
            print(f"Total flights retrieved: {data['pagination']['count']}")
        else:
            print("Warning: Pagination metadata is missing from the API response.")
            
    except requests.exceptions.RequestException as req_err:
        print(f"Network error: Failed to fetch data. Details: {req_err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    fetch_flight_data()