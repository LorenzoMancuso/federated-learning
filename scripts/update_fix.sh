PROXY="lmancuso@130.192.137.199"
CERT_PATH="/home/lore/Scaricati/mancuso_federated"

REMOVE_COMMAND="rm -rf ~/federated-learning"
COMMAND="ssh -o StrictHostKeyChecking=no -i $CERT_PATH -J $PROXY"

# coordinator
rsync -azv --stats --human-readable --info=progress2 --exclude=".*" --exclude="res" -e "$COMMAND" /home/lore/Projects/github/federated-learning/scripts/delete_plus_weights.sh lmancuso@coordinator:/home/lmancuso/federated-learning/Server/snapshots



#rsync -azv -I --stats --human-readable --info=progress2 --exclude=".*" --exclude="res" --exclude="__pycache__" -e "$COMMAND" /home/lore/Projects/federated-learning/Client/ lmancuso@node01:/home/lmancuso/federated-learning/Client/snapshots
#rsync -azv -I --stats --human-readable --info=progress2 --exclude=".*" --exclude="res" --exclude="__pycache__" -e "$COMMAND" /home/lore/Projects/federated-learning/Client/ lmancuso@node02:/home/lmancuso/federated-learning/Client/snapshots
#rsync -azv -I --stats --human-readable --info=progress2 --exclude=".*" --exclude="res" --exclude="__pycache__" -e "$COMMAND" /home/lore/Projects/federated-learning/Client/ lmancuso@node03:/home/lmancuso/federated-learning/Client/snapshots
#rsync -azv -I --stats --human-readable --info=progress2 --exclude=".*" --exclude="res" --exclude="__pycache__" -e "$COMMAND" /home/lore/Projects/federated-learning/Client/ lmancuso@node04:/home/lmancuso/federated-learning/Client/snapshots

rsync -azv -I --stats --human-readable --info=progress2 --exclude=".*" --exclude="res" --exclude="__pycache__" -e "$COMMAND" /home/lore/Projects/github/federated-learning/scripts/delete_plus_weights.sh lmancuso@device1:/home/lmancuso/federated-learning/Client/snapshots
rsync -azv -I --stats --human-readable --info=progress2 --exclude=".*" --exclude="res" --exclude="__pycache__" -e "$COMMAND" /home/lore/Projects/github/federated-learning/scripts/delete_plus_weights.sh lmancuso@device2:/home/lmancuso/federated-learning/Client/snapshots
rsync -azv -I --stats --human-readable --info=progress2 --exclude=".*" --exclude="res" --exclude="__pycache__" -e "$COMMAND" /home/lore/Projects/github/federated-learning/scripts/delete_plus_weights.sh lmancuso@device3:/home/lmancuso/federated-learning/Client/snapshots
rsync -azv -I --stats --human-readable --info=progress2 --exclude=".*" --exclude="res" --exclude="__pycache__" -e "$COMMAND" /home/lore/Projects/github/federated-learning/scripts/delete_plus_weights.sh lmancuso@device4:/home/lmancuso/federated-learning/Client/snapshots
rsync -azv -I --stats --human-readable --info=progress2 --exclude=".*" --exclude="res" --exclude="__pycache__" -e "$COMMAND" /home/lore/Projects/github/federated-learning/scripts/delete_plus_weights.sh lmancuso@device5:/home/lmancuso/federated-learning/Client/snapshots
rsync -azv -I --stats --human-readable --info=progress2 --exclude=".*" --exclude="res" --exclude="__pycache__" -e "$COMMAND" /home/lore/Projects/github/federated-learning/scripts/delete_plus_weights.sh lmancuso@device6:/home/lmancuso/federated-learning/Client/snapshots
rsync -azv -I --stats --human-readable --info=progress2 --exclude=".*" --exclude="res" --exclude="__pycache__" -e "$COMMAND" /home/lore/Projects/github/federated-learning/scripts/delete_plus_weights.sh lmancuso@device7:/home/lmancuso/federated-learning/Client/snapshots
rsync -azv -I --stats --human-readable --info=progress2 --exclude=".*" --exclude="res" --exclude="__pycache__" -e "$COMMAND" /home/lore/Projects/github/federated-learning/scripts/delete_plus_weights.sh lmancuso@device8:/home/lmancuso/federated-learning/Client/snapshots
rsync -azv -I --stats --human-readable --info=progress2 --exclude=".*" --exclude="res" --exclude="__pycache__" -e "$COMMAND" /home/lore/Projects/github/federated-learning/scripts/delete_plus_weights.sh lmancuso@device9:/home/lmancuso/federated-learning/Client/snapshots
rsync -azv -I --stats --human-readable --info=progress2 --exclude=".*" --exclude="res" --exclude="__pycache__" -e "$COMMAND" /home/lore/Projects/github/federated-learning/scripts/delete_plus_weights.sh lmancuso@device10:/home/lmancuso/federated-learning/Client/snapshots
rsync -azv -I --stats --human-readable --info=progress2 --exclude=".*" --exclude="res" --exclude="__pycache__" -e "$COMMAND" /home/lore/Projects/github/federated-learning/scripts/delete_plus_weights.sh lmancuso@device11:/home/lmancuso/federated-learning/Client/snapshots
rsync -azv -I --stats --human-readable --info=progress2 --exclude=".*" --exclude="res" --exclude="__pycache__" -e "$COMMAND" /home/lore/Projects/github/federated-learning/scripts/delete_plus_weights.sh lmancuso@device12:/home/lmancuso/federated-learning/Client/snapshots
rsync -azv -I --stats --human-readable --info=progress2 --exclude=".*" --exclude="res" --exclude="__pycache__" -e "$COMMAND" /home/lore/Projects/github/federated-learning/scripts/delete_plus_weights.sh lmancuso@device13:/home/lmancuso/federated-learning/Client/snapshots
rsync -azv -I --stats --human-readable --info=progress2 --exclude=".*" --exclude="res" --exclude="__pycache__" -e "$COMMAND" /home/lore/Projects/github/federated-learning/scripts/delete_plus_weights.sh lmancuso@device14:/home/lmancuso/federated-learning/Client/snapshots
rsync -azv -I --stats --human-readable --info=progress2 --exclude=".*" --exclude="res" --exclude="__pycache__" -e "$COMMAND" /home/lore/Projects/github/federated-learning/scripts/delete_plus_weights.sh lmancuso@device15:/home/lmancuso/federated-learning/Client/snapshots
rsync -azv -I --stats --human-readable --info=progress2 --exclude=".*" --exclude="res" --exclude="__pycache__" -e "$COMMAND" /home/lore/Projects/github/federated-learning/scripts/delete_plus_weights.sh lmancuso@device16:/home/lmancuso/federated-learning/Client/snapshots
#rsync -azv -I --stats --human-readable --info=progress2 --exclude=".*" --exclude="res" --exclude="__pycache__" -e "$COMMAND" /home/lore/Projects/federated-learning/Client/ lmancuso@device17:/home/lmancuso/federated-learning/Client/snapshots
#rsync -azv -I --stats --human-readable --info=progress2 --exclude=".*" --exclude="res" --exclude="__pycache__" -e "$COMMAND" /home/lore/Projects/federated-learning/Client/ lmancuso@device18:/home/lmancuso/federated-learning/Client/snapshots
#rsync -azv -I --stats --human-readable --info=progress2 --exclude=".*" --exclude="res" --exclude="__pycache__" -e "$COMMAND" /home/lore/Projects/federated-learning/Client/ lmancuso@device19:/home/lmancuso/federated-learning/Client/snapshots
#rsync -azv -I --stats --human-readable --info=progress2 --exclude=".*" --exclude="res" --exclude="__pycache__" -e "$COMMAND" /home/lore/Projects/federated-learning/Client/ lmancuso@device20:/home/lmancuso/federated-learning/Client

#rsync -azv -I --stats --human-readable --info=progress2 --exclude=".*" --exclude="res" --exclude="__pycache__" -e "ssh -o StrictHostKeyChecking=no -i /home/lore/Scaricati/mancuso_federated -J lmancuso@130.192.137.199" /home/lore/Projects/github/federated-learning/Client/ lmancuso@gpu-worker:/home/lmancuso/multi-client/client1/