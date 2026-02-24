print("Script started")
import os
import requests
import time

BASE_DIR = "data/nvidia/raw_html"
os.makedirs(BASE_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "ayisha hala (ayishatp78@gmail.com)"
}

CIK = "0001045810"  # NVIDIA

def download_nvidia_filings():
    print("Fetching NVIDIA filings list...")

    url = f"https://data.sec.gov/submissions/CIK{CIK}.json"
    r = requests.get(url, headers=HEADERS)

    if r.status_code != 200:
        print("Failed to fetch filings.")
        return

    data = r.json()
    filings = data["filings"]["recent"]

    for i in range(len(filings["form"])):
        form = filings["form"][i]
        filing_date = filings["filingDate"][i]
        accession = filings["accessionNumber"][i].replace("-", "")
        primary_doc = filings["primaryDocument"][i]

        if form in ["10-K", "10-Q", "8-K"]:
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(CIK)}/{accession}/{primary_doc}"
            save_path = os.path.join(BASE_DIR, f"{form}_{filing_date}.html")

            try:
                print(f"Downloading {form} - {filing_date}")
                response = requests.get(filing_url, headers=HEADERS)
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(response.text)
                time.sleep(0.5)
            except:
                continue

if __name__ == "__main__":
    download_nvidia_filings()
