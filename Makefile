install:
	pip install -r requirements.txt

train:
	python main.py

image:
	docker build -t your-image-name .

container:
	docker run your-image-name