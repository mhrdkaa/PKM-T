#!/bin/bash
echo "🔍 Mendeteksi port serial dengan baud rate 115200..."
echo "----------------------------------------------"

for port in /dev/tty*; do
    baud=$(stty -F "$port" -a 2>/dev/null | grep -o "speed [0-9]* baud" | awk '{print $2}')
    
    if [ ! -z "$baud" ]; then
        if [ "$baud" -eq 115200 ]; then
            echo "$port → $baud baud"
        else
            echo "$port → $baud baud"
        fi
    fi
done

echo "----------------------------------------------"
echo "Selesai"
