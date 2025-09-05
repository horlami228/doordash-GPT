import os
import csv
import random

# Directory containing your PDFs
PDF_DIR = "data"  # Change this to your directory path

# Base URL (we'll append random IDs to simulate unique links)
BASE_URL = "https://example.com/policies/"

# Output CSV file
OUTPUT_FILE = "pdf_sources.csv"

def generate_random_link():
    """Generate a random link for demonstration purposes."""
    random_id = random.randint(1000, 9999)
    return f"{BASE_URL}{random_id}"

def create_pdf_source_mapping(pdf_dir):
    """Scan PDFs and create filename → random link mapping."""
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]

    with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "source_link"])  # CSV header

        for filename in pdf_files:
            writer.writerow([filename, generate_random_link()])

    print(f"CSV mapping saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    create_pdf_source_mapping(PDF_DIR)
