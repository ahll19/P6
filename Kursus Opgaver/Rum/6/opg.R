library("spatstat")
# (1)
#   Jeg ville gerne bruge en Thomas Process
#   omega giver os større spredning når den selv bliver større
#   Kappa er middel antal earthquakes
#   alpha er middel antal efterrystelser ved et givet earthquake
#
#   vi mangler intensiteten på den homogene center process

# (2)
#   cox: generationer af svampe baseret på hvor deres sporer flyver hen
#   poisson: noget der ikke ændrer sig over tid eller har en tilfældig intensitet lmao

# (4)
#  (a)
plot(redwood)
matern1 <- rMatClust(10, 0.05, 5)
plot(matern1)
thomas1 <- rThomas(10, 0.05, 5)
plot(thomas1)
#  (b)
thin <- function (x, y, r, x0, y0){
  ifelse ((x-x0)^2+(y-y0)^2<r^2, 0, 1)
}
nclust <- function (x0, y0, r1, r2, beta){
  num <- rpois(1, beta)
  xc <- runifdisc(num, r2, centre = c(x0, y0))
  xc <- rthin(xc, thin, r = r1, x0 = x0, y0 = y0)
  return(xc)
}
neymannGeneral <- rNeymanScott(10, 10, nclust, r1 = 0, r2 = 2, beta = 10)
plot(neymannGeneral)

# (5)
#   Se noter

# (6)
#  (a)
model1 <- kppm(redwood, ~1, clusters="Thomas", statistic="pcf")
model1
model1$Fit
#  (b)
redwoodEstimation1 <- rThomas(25.29693932, 0.03968141, 2.450889)
plot(redwood)
plot(redwoodEstimation1)
#  (c)
model2 <- kppm(redwood, ~1, clusters="Thomas", statistic="K")
redwoodEstimation2 <- rThomas(23.54856848, 0.04705148, 2.632856)
plot(redwood)
plot(redwoodEstimation2)
#  (d)
modelMatern <- kppm(redwood, ~1, clusters="MatClust", statistic = "pcf")
redwoodMaternFit <- rMatClust(24.34099863, 0.07873234, 2.547143)
plot(redwood)
plot(redwoodMaternFit)