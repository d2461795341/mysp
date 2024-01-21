CURRENT_DIR=$(pwd)

cd data/mit-states/images

for dir in */; do
    new_dir=$(echo "$dir" | tr ' ' '_')
    mv "$dir" "$new_dir"
done

cd "$CURRENT_DIR"
