import torch
import numpy as np

bikes_numpy = np.loadtxt(
    "../../../../dlwpt-code/data/p1ch4/bike-sharing-dataset/hour-fixed.csv",
    dtype=np.float32,
    delimiter=",",
    skiprows=1,
    # 2010-01-10 --> 10
    converters={1:lambda x: float(x[8:10])})
bikes = torch.from_numpy(bikes_numpy)
print(bikes)
print(bikes.shape, " bikes.shape")
print(bikes.stride(), " bikes.stride()")

daily_bikes = bikes.view(-1, 24, bikes.shape[1])
print(daily_bikes.shape, daily_bikes.stride())

# transpose
daily_bikes = daily_bikes.transpose(1, 2)
print(daily_bikes.shape, daily_bikes.stride())

# one hot
first_day = bikes[:24].long()
weather_onehot = torch.zeros(first_day.shape[0], 4)
# weather
print(first_day[:, 9], " weather leval")
weather_onehot.scatter_(
    dim=1,
    index=first_day[:, 9].unsqueeze(1).long() - 1,
    value=1.0
)

print(weather_onehot, " weather_onehot")

# cancatenate
cat_weather_onehot = torch.cat([bikes[:24], weather_onehot], dim=1)[:1]
print(cat_weather_onehot.shape, " cat_weather_onehot.shape")

daily_weather_onehot = torch.zeros(daily_bikes.shape[0], 4, daily_bikes.shape[2])
print(daily_weather_onehot.shape, " daily_weather_onehot.shape")
daily_weather_onehot = daily_weather_onehot.scatter_(
    dim=1,
    index = daily_bikes[:, 9, :].unsqueeze(1).long() - 1,
    value = 1.0
)

daily_weather_onehot = torch.cat([daily_bikes, daily_weather_onehot], dim=1)
print(daily_weather_onehot, " daily_weather_onehot")

# rather than onehot, normalizing weather is another choice
daily_bikes[:, 9, :] = (daily_bikes[:, 9, :] - 1.0) / 3.0

# nomalizing temp
temp = daily_bikes[:, 10, :]
temp_max = torch.max(temp)
temp_min = torch.min(temp)
daily_bikes[:, 10, :] = (daily_bikes[:, 10, :] - temp_min) / (temp_max - temp_min)

# substract the mean and divide by the standard deviation.
temp = daily_bikes[:, 10, :]
daily_bikes[:, 10, :] = ( daily_bikes[:, 10, :] - torch.mean(temp) ) / torch.std(temp)
