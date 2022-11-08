from locust import HttpUser, task


IMAGE = "/home/mykola/kuna-ml/yolov5/data/images/zidane.jpg"
with open(IMAGE, "rb") as f:
    image_data = f.read()


class HelloWorldUser(HttpUser):
    @task
    def hello_world(self):
        self.client.post("/v1/object-detection/yolov5s", files={"image": image_data})
        
