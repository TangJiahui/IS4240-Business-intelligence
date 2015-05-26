#!/usr/bin/env python3

import os
import textblob
import csv

f = os.listdir()

for filename in f:
    print(filename)
    with open(filename, 'r', encoding="latin-1") as csvfile:
        with open(filename+".sentiment.csv", "w") as output:
            reader = csv.DictReader(csvfile)
            writer = csv.DictWriter(output, fieldnames=["text", "post_date", "sentiment"], extrasaction='ignore')
            for row in reader:
                text = row['text']
                blob = textblob.TextBlob(text)
                try:
                    lan = blob.detect_language()
                except:
                    lan = ''
                if lan == 'en':
                    row['sentiment'] = blob.sentiment.polarity
                    writer.writerow(row)
                