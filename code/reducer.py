# Reducer
import sys

current_airline = None
flight_count = 0

for line in sys.stdin:
    # Parse the input from the Mapper
    airline, count = line.strip().split('\t')

    # Convert count to an integer
    count = int(count)

    # Check if the airline has changed
    if current_airline != airline:
        # Output the result for the previous airline
        if current_airline:
            print(f"{current_airline}\t{flight_count}")

        # Reset count for the new airline
        current_airline = airline
        flight_count = count
    else:
        # Increment count for the same airline
        flight_count += count

# Output the result for the last airline
if current_airline:
    print(f"{current_airline}\t{flight_count}")
