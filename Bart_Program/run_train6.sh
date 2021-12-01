# /bin/bash
INSPUR1="srun -p inspur -w inspur-gpu-01" 
INSPUR1="srun --gres=gpu:V100:1 -x dell-gpu-08" 


sample=0.1
comment="mtl_rel_cbr_infill+2+_real_smp_${sample}_revise"
log_file="./Bart_Program/error_log/${comment}.log"
log_content=`cat $log_file`
if [[ ! -f $log_file ]] || [[ $log_content =~ "CUDA out of memory" ]] ; then
    echo "ok"
else
    echo "fail"
fi