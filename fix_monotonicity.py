import csv

def patch_submission(input_file="submission.csv", output_file="submission_fixed.csv"):
    with open(input_file, 'r') as infile:
        reader = list(csv.DictReader(infile))
        
    fieldnames = ['candidate_id', 'rank', 'score', 'reasoning']
    
    # Ensure the first row's score remains untouched
    current_score = float(reader[0]['score'])
    
    for i in range(1, len(reader)):
        score = float(reader[i]['score'])
        
        # If this score is greater than or equal to the rank above it, force it lower
        if score >= current_score:
            score = current_score - 0.0001
            
        reader[i]['score'] = f"{score:.4f}"
        current_score = score

    with open(output_file, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(reader)
        
    print(f"✅ Successfully patched! Saved as {output_file}")

if __name__ == "__main__":
    patch_submission()