mv lcdreadme-.txt README
for year in `seq 1959 2023` ; do
	unzip ${year}*zip
        mkdir -p $year
        mv *lcd* $year/
        mv *LCD* $year/
        cd $year/

	if [ $year -lt 1997 ] ; then
        	for file in *lcd* ; do
		  newfile=${file:3:4}
		  newfile=`echo ${newfile}".lcd"`
		  mv $file $newfile
		done
	
	elif [ $year -ge 1997 ] && [ $year -lt 2000 ]; then
		for file in *LCD* ; do
       	 	  newfile=${file:0:4}
        	  newfile=`echo ${newfile}".lcd"`
      		  mv $file $newfile
		  done

	elif [ $year -eq 2023 ] ; then
		rm -rf *xlsx *.mgr *climatology TAFs

	fi
        cd ../
done
rm -f datapackage.json *.zip
mv README lcdreadme-.txt
