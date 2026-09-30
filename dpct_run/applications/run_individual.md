
 
## CartPole

### Legacy .properties config examples

```sh
dpct-run-individual samples/configs/legacy/CartPole-6150fd-reward.properties --run --early-termination --render
dpct-run-individual samples/configs/legacy/CartPole-6150fd-reward.properties --save-config
dpct-run-individual samples/configs/legacy/CartPole-6150fd-reward.properties --run --steps 500 --seed 1 --iterations 3
```

## LunarLander

### Legacy .properties config examples

```sh
dpct-run-individual samples/configs/legacy/LunarLander-4905d2.properties --run --early-termination --render
dpct-run-individual samples/configs/legacy/LunarLander-4905d2.properties --network-image /tmp/images/lunar_network.png
dpct-run-individual samples/configs/legacy/LunarLander-4905d2.properties --run --early-termination --seed 1 --iterations 100
dpct-run-individual samples/configs/legacy/LunarLander-4905d2.properties --network-image /tmp/lunar/lunar_network.png --save-config

dpct-run-individual samples/configs/legacy/LunarLander-c4d6de.properties --run --early-termination --render
dpct-run-individual samples/configs/legacy/LunarLander-2e0e6f.properties --run --render --early-termination
dpct-run-individual samples/configs/legacy/LunarLander-a420c93.properties --run --render --early-termination
dpct-run-individual samples/configs/legacy/LunarLander-7666b6.properties --run --render --early-termination
dpct-run-individual samples/configs/legacy/LunarLander-f7d501.properties --run --render --early-termination
dpct-run-individual samples/configs/legacy/LunarLander-82e7b5.properties --run --render --early-termination
```

### DPCT .json config examples

```sh
dpct-run-individual samples/configs/dpct/LunarLander-7666b6.json --run --early-termination --render
dpct-run-individual samples/configs/dpct/LunarLander-4905d2.json --run --early-termination --render
```

## MountainCarContinuous

### Legacy .properties config examples

```sh
dpct-run-individual samples/configs/legacy/MountainCarContinuous-cdf7cc.properties --run --early-termination --render
dpct-run-individual samples/configs/legacy/MountainCarContinuous-cdf7cc.properties --show-config --save-config
dpct-run-individual samples/configs/legacy/MountainCarContinuous-cdf7cc.properties --network-image /tmp/images/mcc_network.png --network-line-thickness 2.0 --network-label-fontsize 12 --network-weight-fontsize 10
dpct-run-individual samples/configs/legacy/MountainCarContinuous-cdf7cc.properties --run --render --early-termination --seed 1
dpct-run-individual samples/configs/legacy/MountainCarContinuous-cdf7cc.properties --run --iterations 100 --early-termination --seed 1

dpct-run-individual samples/configs/legacy/MountainCarContinuous-cdf7cc.properties --run --steps 999 --seed 1 --history-dir /tmp/dpct-run-individual-history1 --history-graphs actions,observations,level0-output,reference-perception --early-termination --network-image /tmp/dpct-run-individual-history1/mcc_network.png --render

```

### DPCT .json config examples

```sh
dpct-run-individual samples/configs/dpct/MountainCarContinuous-cdf7cc.json --run --steps 999 --iterations 5 --seed 42
dpct-run-individual samples/configs/dpct/MountainCarContinuous-cdf7cc.json --run --steps 999 --seed 1 --history-dir /tmp/dpct-run-individual-history --history-graphs actions,observations,level0-output,reference-perception --early-termination --network-image /tmp/dpct-run-individual-history/mcc_network.png --network-line-thickness 2.0 --network-label-fontsize 12 --network-weight-fontsize 10 --render

dpct-run-individual https://www.comet.com/mountaincarcontinuous-v0/rms/36f83e3334f3408ea1c473d232d87efe --run --steps 999 --seed 1 --history-dir /tmp/dpct-comet-history/36f83e3334f3408ea1c473d232d87efe  --history-graphs actions,observations,level0-output,reference-perception  --early-termination --network-image /tmp/dpct-comet-history/36f83e3334f3408ea1c473d232d87efe/mcc_network.png --render

dpct-run-individual https://www.comet.com/mountaincarcontinuous-v0/rms/36f83e3334f3408ea1c473d232d87efe --run --steps 999 --seed 3554922200 --history-dir /tmp/dpct-comet-history/36f83e3334f3408ea1c473d232d87efe  --history-graphs actions,observations,level0-output,reference-perception  --early-termination --network-image /tmp/dpct-comet-history/36f83e3334f3408ea1c473d232d87efe/mcc_network.png --render

dpct-run-individual https://www.comet.com/mountaincarcontinuous-v0/rms-top/82355a953b954738853ae2d544ccc221 --run --steps 999 --render  --early-termination 

dpct-run-individual https://www.comet.com/mountaincarcontinuous-v0/rms-top/82355a953b954738853ae2d544ccc221 --run --steps 999 --early-termination --seed 1 --iterations 5


dpct-run-individual https://www.comet.com/mountaincarcontinuous-v0/rms/8e3696420a52459f8da99f22e2deb5ad --run --steps 999 --early-termination 

dpct-run-individual https://www.comet.com/mountaincarcontinuous-v0/rms/36f83e3334f3408ea1c473d232d87efe --run --steps 999 --early-termination --render
dpct-run-individual https://www.comet.com/mountaincarcontinuous-v0/cumulative-reward/3d89bbfca24c4815967c32f42a0fc6f3 --run --steps 999 --render  --early-termination 

dpct-run-individual https://www.comet.com/mountaincarcontinuous-v0/evaluation-steps/a3719e34e58e40c2a474b5049e2dfb57 --run --steps 999 --render  --early-termination --history-graphs actions,observations,level0-output,reference-perception  --history-dir /tmp/dpct-run-individual-history --network-image /tmp/dpct-run-individual-history/mcc_network.png

dpct-run-individual https://www.comet.com/mountaincarcontinuous-v0/evaluation-steps/a3719e34e58e40c2a474b5049e2dfb57 --run --steps 999 --render  --early-termination --network-image /tmp/dpct-run-individual-history/mcc_network.png 

dpct-run-individual https://www.comet.com/mountaincarcontinuous-v0/evaluation-steps/a3719e34e58e40c2a474b5049e2dfb57 --network-image /tmp/dpct-run-individual-history/mcc_network.png --network-line-thickness 5 --network-label-fontsize 16 --network-weight-fontsize 14 --network-smooth-factor-fontsize 12 --network-bottom-label-fontsize 14 --network-legend-fontsize 16 --network-legend-title-fontsize 15 

dpct-run-individual https://www.comet.com/mountaincarcontinuous-v0/evaluation-steps/a3719e34e58e40c2a474b5049e2dfb57 --network-image /tmp/dpct-run-individual-history/mcc_network.png 


dpct-run-individual https://www.comet.com/mountaincarcontinuous-v0/rms-top/9730fe84423d49398f3ee09c6051955e --run --steps 999 --render  --early-termination 




```

### Video export examples

```sh
dpct-run-individual https://www.comet.com/mountaincarcontinuous-v0/evaluation-steps/a3719e34e58e40c2a474b5049e2dfb57 --run --steps 999 --early-termination --seed 1 --video-out /tmp/dpct-run-individual-history/mcc_rollout.mp4 --video-fps 30

dpct-run-individual https://www.comet.com/mountaincarcontinuous-v0/evaluation-steps/a3719e34e58e40c2a474b5049e2dfb57 --run --steps 999 --early-termination --video-out /tmp/dpct-comet-history/eval_steps.gif --video-fps 20
```




