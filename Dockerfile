FROM python:3.10-slim
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["python", "./sir_label.py", "> /datasets/SIR_Results/output_BA_machine_log.log 2>&1"]
# MACHINE_NAME=negin_mch
# RESULT_ADDRESS=./datasets/SIR_Results/
# DATASET_DIR=./datasets/
# # POOL_SIZE=1
# POOL_SIZE=4
# # POOL_SIZE=6