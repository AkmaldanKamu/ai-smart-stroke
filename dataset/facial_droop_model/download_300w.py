import deeplake
ds = deeplake.load("hub://activeloop/300w")
print(ds.info)