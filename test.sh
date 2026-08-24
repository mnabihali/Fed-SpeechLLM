#!/bin/bash

CSV_FILES=(
    "/stek/mohamed/FL-SLAM/metween_manifest.csv"
    "/stek/mohamed/FL-SLAM/test.csv"
    "/stek/mohamed/SpeechLLM/csv_mlsitalian/test_itmls.csv"
    "/stek/mohamed/SpeechLLM/csv_mlsfrench/test_frmls.csv"
    "/stek/mohamed/SpeechLLM/csv_mlsgerman/test_demls.csv"
    "/stek/mohamed/SpeechLLM/csv_mlsspanish/test_esmls.csv"
    "/stek/mohamed/SpeechLLM/csv_mlsdutch/test_nlmls.csv"
)

for csv in "${CSV_FILES[@]}"; do
    echo "Running inference on: $csv"

    logfile="temp.log"

    # Show progress + save output
    python test.py --round 60 --csv "$csv" 2>&1 | tee "$logfile"

    # Extract WER
    wer=$(grep "val/wer" "$logfile" | tail -n 1 | awk '{print $3}')

    echo "$(basename $csv) ? WER: $wer"
    echo "----------------------------------"

    # Append to main log
    cat "$logfile" >> inference-serverfinealllang-60.log
done

echo "All CSVs processed!"
##!/bin/bash
#
#CSV_FILES=(
#    "/stek/mohamed/FL-SLAM/test.csv"
#    "/stek/mohamed/SpeechLLM/csv_mlsitalian/test_itmls.csv"
#    "/stek/mohamed/SpeechLLM/csv_mlsfrench/test_frmls.csv"
#    "/stek/mohamed/SpeechLLM/csv_mlsgerman/test_demls.csv"
#    "/stek/mohamed/SpeechLLM/csv_mlsspanish/test_esmls.csv"
#    "/stek/mohamed/SpeechLLM/csv_mlsdutch/test_nlmls.csv"
#)
#
#for csv in "${CSV_FILES[@]}"; do
#    echo "Running inference on: $csv"
#    python test.py --round 18 --csv "$csv" 2>&1 | tee -a inference-18.log
#    echo "Done with: $csv"
#done
#
#echo "All CSVs processed!"
