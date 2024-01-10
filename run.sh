tmux new -d -s TCA -n window0
tmux split-window -h -t TCA:window0
tmux split-window -h -t TCA:window0
tmux split-window -h -t TCA:window0
tmux split-window -v -t TCA:window0.0
tmux split-window -v -t TCA:window0.1
tmux split-window -v -t TCA:window0.2
tmux split-window -v -t TCA:window0.3
tmux split-window -v -t TCA:window0.4
tmux split-window -v -t TCA:window0.5
tmux split-window -v -t TCA:window0.6
tmux split-window -v -t TCA:window0.7
tmux split-window -v -t TCA:window0.8
tmux split-window -v -t TCA:window0.9
tmux split-window -v -t TCA:window0.10
tmux split-window -v -t TCA:window0.11
tmux split-window -v -t TCA:window0.12
tmux split-window -v -t TCA:window0.13
tmux split-window -v -t TCA:window0.14

dataset=(
  "TIDIGITS"
	"MedleyDB"
)

for ((j=0;j<2;j+=1))
do
  for ((i=0;i<8;i+=1))
  do
    k=$(($i + $j * 8))
    tmux send -t TCA:window0.$k "conda activate TCA" ENTER
    tmux send -t TCA:window0.$k "python main.py --dataset ${dataset[$j]} --seed $i --gpu $(($k%3))" ENTER
  done
done

tmux a -t TCA