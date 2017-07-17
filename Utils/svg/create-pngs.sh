#! /bin/bash
# create .pngs for svg dataset

# progress updates every N files
N=100

find dataset -type d | while read sketch_dir
do
	
	# maybe clean up directory
	# rm "$sketch_dir"/*.svg
	# rm "$sketch_dir"/*.png

	# remove temporary files
	# rm $(ls "$sketch_dir/" | grep ~)

	# get list of pngs
	list_of_pngs=$(ls "$sketch_dir" | grep .png)
	#echo $list_of_pngs: ${#list_of_pngs[@]}

	# check for pngs and generate them if not found
	if [[ "$list_of_pngs" == "" ]]; then
	    gtimeout 5s mogrify -resize 256x256 -format png -- "$sketch_dir"/*.svg
	fi

	for png_path in $list_of_pngs
	do
		full_path="$sketch_dir"/"$png_path"
		width=$(identify -format "%w" "$full_path")
		echo size: $width
		#rm "$full_path"
		# if [[ -f "$full_path" && "$width" != "225" ]]
		# then
		# 	echo rezising...
		# 	mogrify "$full_path" -resize 225x225! "$full_path"
		# else
		# 	
		# fi
	done
	((i=i%N))
	((i=i+1))
	if [ "$i" -eq "$N" ]; then
	    echo $sketch_dir ...done!
	fi
	
done
