PROXY="lmancuso@130.192.137.199"
CERT_PATH="/home/lore/Scaricati/mancuso_federated"

# 2) copia dei file richiesti tramite sync
COMMAND="ssh -i $CERT_PATH -J $PROXY"


#rsync -azv --append-verify --stats --human-readable --info=progress2 -e "ssh -i /home/lore/Scaricati/mancuso_federated" /media/lore/B6C8D9F4C8D9B33B/Users/lorym/Downloads/IMAGENET/1597433771/subset02/ hpce17612@plogin1.bsc.es:/nvme1/imagenet/
#rsync -avzP username@dt01.bsc.es:remotefile_or_remotedir localdir
#
#rsync -azv --ignore-existing --remove-source-files --stats --human-readable --info=progress2 `ls | head -258`  /gpfs/home/hpce17/hpce17612/imagenet_34ss


# validation set coordinator
#rsync -azv --append-verify --stats --human-readable --info=progress2 -e "$COMMAND" /media/lore/EA72A48772A459D9/ILSVRC2012/ILSVRC2012_img_val lmancuso@coordinator:~/federated-learning/Server/res/
#rsync -azv --append-verify --stats --human-readable --info=progress2 -e "$COMMAND" /home/lore/Projects/ILSVRC2012_devkit_t12 lmancuso@coordinator:~/federated-learning/Server/res/

#echo 'clean device1'
#ssh -i $CERT_PATH -J $PROXY lmancuso@device1 "rm -rf /home/lmancuso/dataset/subset/*"
#echo 'clean device2'
#ssh -i $CERT_PATH -J $PROXY lmancuso@device2 "rm -rf /home/lmancuso/dataset/subset/*"
#echo 'clean device3'
#ssh -i $CERT_PATH -J $PROXY lmancuso@device3 "rm -rf /home/lmancuso/dataset/subset/*"
#echo 'clean device4'
#ssh -i $CERT_PATH -J $PROXY lmancuso@device4 "rm -rf /home/lmancuso/dataset/subset/*"
#echo 'clean device5'
#ssh -i $CERT_PATH -J $PROXY lmancuso@device5 "rm -rf /home/lmancuso/dataset/subset/*"
#echo 'clean device6'
#ssh -i $CERT_PATH -J $PROXY lmancuso@device6 "rm -rf /home/lmancuso/dataset/subset/*"
echo 'clean device7'
ssh -i $CERT_PATH -J $PROXY lmancuso@device7 "rm -rf /home/lmancuso/dataset/subset/*"
echo 'clean device8'
ssh -i $CERT_PATH -J $PROXY lmancuso@device8 "rm -rf /home/lmancuso/dataset/subset/*"
echo 'clean device9'
ssh -i $CERT_PATH -J $PROXY lmancuso@device9 "rm -rf /home/lmancuso/dataset/subset/*"
echo 'clean device10'
ssh -i $CERT_PATH -J $PROXY lmancuso@device10 "rm -rf /home/lmancuso/dataset/subset/*"
echo 'clean device11'
ssh -i $CERT_PATH -J $PROXY lmancuso@device11 "rm -rf /home/lmancuso/dataset/subset/*"
echo 'clean device12'
ssh -i $CERT_PATH -J $PROXY lmancuso@device12 "rm -rf /home/lmancuso/dataset/subset/*"
echo 'clean device13'
ssh -i $CERT_PATH -J $PROXY lmancuso@device13 "rm -rf /home/lmancuso/dataset/subset/*"
ssh -i $CERT_PATH -J $PROXY lmancuso@device13 "rm -rf /home/lmancuso/dataset/subset*"
echo 'clean device14'
ssh -i $CERT_PATH -J $PROXY lmancuso@device14 "rm -rf /home/lmancuso/dataset/subset/*"
ssh -i $CERT_PATH -J $PROXY lmancuso@device14 "rm -rf /home/lmancuso/dataset/subset*"
echo 'clean device15'
ssh -i $CERT_PATH -J $PROXY lmancuso@device15 "rm -rf /home/lmancuso/dataset/subset/*"
ssh -i $CERT_PATH -J $PROXY lmancuso@device15 "rm -rf /home/lmancuso/dataset/subset*"
echo 'clean device16'
ssh -i $CERT_PATH -J $PROXY lmancuso@device16 "rm -rf /home/lmancuso/dataset/subset/*"
ssh -i $CERT_PATH -J $PROXY lmancuso@device16 "rm -rf /home/lmancuso/dataset/subset*"


# CPU DEVICES
#echo 'device1'
#rsync -azv --append-verify --stats --human-readable --info=progress2 -e "$COMMAND" /media/lore/B6C8D9F4C8D9B33B/Users/lorym/Downloads/IMAGENET/1597433771/subset01/ lmancuso@device1:/home/lmancuso/dataset/subset/
#echo 'device2'
#rsync -azv --append-verify --stats --human-readable --info=progress2 -e "$COMMAND" /media/lore/B6C8D9F4C8D9B33B/Users/lorym/Downloads/IMAGENET/1597433771/subset02/ lmancuso@device2:/home/lmancuso/dataset/subset/
#echo 'device3'
#rsync -azv --append-verify --stats --human-readable --info=progress2 -e "$COMMAND" /media/lore/B6C8D9F4C8D9B33B/Users/lorym/Downloads/IMAGENET/1597433771/subset03/ lmancuso@device3:/home/lmancuso/dataset/subset/
#echo 'device4'
#rsync -azv --append-verify --stats --human-readable --info=progress2 -e "$COMMAND" /media/lore/B6C8D9F4C8D9B33B/Users/lorym/Downloads/IMAGENET/1597433771/subset04/ lmancuso@device4:/home/lmancuso/dataset/subset/
#echo 'device5'
#rsync -azv --append-verify --stats --human-readable --info=progress2 -e "$COMMAND" /media/lore/B6C8D9F4C8D9B33B/Users/lorym/Downloads/IMAGENET/1597433771/subset05/ lmancuso@device5:/home/lmancuso/dataset/subset/
#echo 'device6'
#rsync -azv --append-verify --stats --human-readable --info=progress2 -e "$COMMAND" /media/lore/B6C8D9F4C8D9B33B/Users/lorym/Downloads/IMAGENET/1597433771/subset06/ lmancuso@device6:/home/lmancuso/dataset/subset/
echo 'device7'
rsync -azv --append-verify --stats --human-readable --info=progress2 -e "$COMMAND" /media/lore/B6C8D9F4C8D9B33B/Users/lorym/Downloads/IMAGENET/1597433771/subset07/ lmancuso@device7:/home/lmancuso/dataset/subset/
echo 'device8'
rsync -azv --append-verify --stats --human-readable --info=progress2 -e "$COMMAND" /media/lore/B6C8D9F4C8D9B33B/Users/lorym/Downloads/IMAGENET/1597433771/subset08/ lmancuso@device8:/home/lmancuso/dataset/subset/
echo 'device9'
rsync -azv --append-verify --stats --human-readable --info=progress2 -e "$COMMAND" /media/lore/B6C8D9F4C8D9B33B/Users/lorym/Downloads/IMAGENET/1597433771/subset09/ lmancuso@device9:/home/lmancuso/dataset/subset/
echo 'device10'
rsync -azv --append-verify --stats --human-readable --info=progress2 -e "$COMMAND" /media/lore/B6C8D9F4C8D9B33B/Users/lorym/Downloads/IMAGENET/1597433771/subset10/ lmancuso@device10:/home/lmancuso/dataset/subset/
echo 'device11'
rsync -azv --append-verify --stats --human-readable --info=progress2 -e "$COMMAND" /media/lore/B6C8D9F4C8D9B33B/Users/lorym/Downloads/IMAGENET/1597433771/subset11/ lmancuso@device11:/home/lmancuso/dataset/subset/
echo 'device12'
rsync -azv --append-verify --stats --human-readable --info=progress2 -e "$COMMAND" /media/lore/B6C8D9F4C8D9B33B/Users/lorym/Downloads/IMAGENET/1597433771/subset12/ lmancuso@device12:/home/lmancuso/dataset/subset/
# dataset device13
echo 'device13'
rsync -azv --append-verify --stats --human-readable --info=progress2 -e "$COMMAND" /media/lore/B6C8D9F4C8D9B33B/Users/lorym/Downloads/IMAGENET/1597433771/subset13/ lmancuso@device13:/home/lmancuso/dataset/subset/
# dataset device14
echo 'device14'
rsync -azv --append-verify --stats --human-readable --info=progress2 -e "$COMMAND" /media/lore/B6C8D9F4C8D9B33B/Users/lorym/Downloads/IMAGENET/1597433771/subset14/ lmancuso@device14:/home/lmancuso/dataset/subset/
# dataset device15
echo 'device15'
rsync -azv --append-verify --stats --human-readable --info=progress2 -e "$COMMAND" /media/lore/B6C8D9F4C8D9B33B/Users/lorym/Downloads/IMAGENET/1597433771/subset15/ lmancuso@device15:/home/lmancuso/dataset/subset/
# dataset device16
echo 'device16'
rsync -azv --append-verify --stats --human-readable --info=progress2 -e "$COMMAND" /media/lore/B6C8D9F4C8D9B33B/Users/lorym/Downloads/IMAGENET/1597433771/subset16/ lmancuso@device16:/home/lmancuso/dataset/subset/




## dataset device17
#echo 'device17'
##ssh -i $CERT_PATH -J $PROXY lmancuso@device4 "rm -rf /home/lmancuso/dataset/subset*"
#rsync -azv --append-verify --stats --human-readable --info=progress2 -e "$COMMAND" /media/lore/B6C8D9F4C8D9B33B/Users/lorym/Downloads/IMAGENET/1597433771/subset21 lmancuso@device17:/home/lmancuso/dataset/
#
## dataset device18
#echo 'device18'
##ssh -i $CERT_PATH -J $PROXY lmancuso@device4 "rm -rf /home/lmancuso/dataset/subset*"
#rsync -azv --append-verify --stats --human-readable --info=progress2 -e "$COMMAND" /media/lore/B6C8D9F4C8D9B33B/Users/lorym/Downloads/IMAGENET/1597433771/subset22 lmancuso@device18:/home/lmancuso/dataset/
#
## dataset device19
#echo 'device19'
##ssh -i $CERT_PATH -J $PROXY lmancuso@device4 "rm -rf /home/lmancuso/dataset/subset*"
#rsync -azv --append-verify --stats --human-readable --info=progress2 -e "$COMMAND" /media/lore/B6C8D9F4C8D9B33B/Users/lorym/Downloads/IMAGENET/1597433771/subset23 lmancuso@device19:/home/lmancuso/dataset/
#
## dataset device20
#echo 'device20'
##ssh -i $CERT_PATH -J $PROXY lmancuso@device4 "rm -rf /home/lmancuso/dataset/subset*"
#rsync -azv --append-verify --stats --human-readable --info=progress2 -e "$COMMAND" /media/lore/B6C8D9F4C8D9B33B/Users/lorym/Downloads/IMAGENET/1597433771/subset24 lmancuso@device20:/home/lmancuso/dataset/
#


rsync -azv --bwlimit=2000 --append-verify --stats --human-readable --info=progress2 -e "ssh -i /home/lore/Scaricati/mancuso_federated -J lmancuso@130.192.137.199" /media/lore/B6C8D9F4C8D9B33B/Users/lorym/Downloads/IMAGENET/1597433771/subset1 lmancuso@gpu-worker:/mnt/dataset/IMAGENET/
