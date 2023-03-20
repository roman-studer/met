FROM tensorflow/tensorflow

# install git
RUN apt-get update && apt-get install -y git python3-pip
RUN pip3 install --upgrade pip

# install python packages
COPY ./requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r ./requirements.txt
