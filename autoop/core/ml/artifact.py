from pydantic import BaseModel, Field
import base64

class Artifact(BaseModel):
    pass



# Has methods:
# .__init__(type="dataset", *args, **kwargs)
# super().read() --> some kind bytes type
# super().save(bytes)
