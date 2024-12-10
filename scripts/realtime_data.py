import oandapyV20
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.instruments as instruments
import time

# Set your access token and account details
ACCOUNT_ID = "123-456-78901234-012"
ACCESS_TOKEN = "abcd-efgh"

# Initialize the OANDA client
client = oandapyV20.API(access_token=ACCESS_TOKEN, environment="live")  # Use "practice" for demo accounts

# Function to fetch live pricing data
def fetch_live_pricing(account_id, inst):
    params = {
        "instruments": inst,  # e.g., "USD_JPY,EUR_USD"
    }
    params_vol = {
        "granularity": "S5",  # e.g., "M1" for 1-minute data
        "count": 1,  
    }
    r = pricing.PricingInfo(accountID=account_id, params=params)
    r_vol = instruments.InstrumentsCandles(instrument=inst, params=params_vol)
    try:
        response = client.request(r)
        response_vol = client.request(r_vol)

        candles = response_vol.get("candles", [])
        prices = response.get("prices", [])

        instrument = prices[0].get("instrument", "Unknown")
        bid = prices[0].get("bids", [{}])[0].get("price", "N/A")
        ask = prices[0].get("asks", [{}])[0].get("price", "N/A")
        volume = candles[0].get("volume", "N/A")
        print(f"{instrument}: Bid={bid}, Ask={ask}, Volume={volume}")
    except Exception as e:
        print("Error fetching pricing data:", e)

# Poll live pricing every 5 seconds
while True:
    fetch_live_pricing(ACCOUNT_ID, "USD_JPY")  # Replace with other instruments as needed
    time.sleep(5)
