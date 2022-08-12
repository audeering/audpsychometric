

datasetdir <- "/home/audeering.local/cgeng/code/tools/audnoname/audnoname/core/datasets/"
fname <- paste(datasetdir, "HolzingerSwineford1939.csv", sep="")
data <- read.csv(fname, header = TRUE)

model <- "g =~ x1+ x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9"
fit <- lavaan::cfa(model = model, data = data)
model_loadings <- lavaan::inspect(fit, what = "std")[["lambda"]]
print(model_loadings)


print(fit)

loading_sum <- sum(model_loadings) ** 2
