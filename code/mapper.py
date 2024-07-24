# Mapper
import sys

for line in sys.stdin:
    # Assuming each line is a CSV record
    record = line.strip().split(',')

    # Assuming the Airline column is at index 0
    airline = record[0]

    # Emit key-value pair: (airline, 1)
    print(f"{airline}\t1")
