#!/bin/bash
for filename in *.jpg; do  
  error=`identify -regard-warnings $filename 2>&1 >/dev/null;`
  if [ $? -ne 0 ]; then
    echo "The image is corrupt or unknown format"
    echo "$error"
    echo $filename >> $1
  fi
done