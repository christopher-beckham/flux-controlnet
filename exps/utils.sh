#!/bin/bash

split_string_with_slash() {
    local input="$1"
    local a b
    if [[ "$input" == */* ]]; then
        a=$(echo "$input" | awk -F'/' '{print $1}')
        b=$(echo "$input" | awk -F'/' '{print $2}')
    else
        a="$input"
        b=""
    fi
    echo "$a" "$b"
}

export -f split_string_with_slash

alias gitgrep="git log -p --all -S"
