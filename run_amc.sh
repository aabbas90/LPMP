# First run the instance which is working correctly:
echo "------------RUNNING A SUCCESSFULLY SOLVED INSTANCE (takes around 70secs on MPI machine) ------------"
python run_amc.py /BS/ahmed_projects/work/data/amc_infinite_loop/running_instance_amc.pkl

echo "------------RUNNING THE PROBLEMATIC INSTANCE (RAM will keep growing)------------"
python run_amc.py /BS/ahmed_projects/work/data/amc_infinite_loop/failing_instance_amc.pkl