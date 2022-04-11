library(spatstat)
library(evmix)
library(RandomFields)
m <- split(bramblecanes)
year0 <- m$'0'; year1 <- m$'1'; year2 <- m$'2'

lest0 <- Lest(year0); plot(lest0, .-r~r)
lest1 <- Lest(year1); plot(lest1, .-r~r)
lest2 <- Lest(year2); plot(lest2, .-r~r)

year10 <- Lcross(bramblecanes, '1', '0'); plot(year10, .-r~r)
year21 <- Lcross(bramblecanes, '2', '1'); plot(year21, .-r~r)
year20 <- Lcross(bramblecanes, '2', '0'); plot(year20, .-r~r)

# log gaus
# Det er åbenbart standard at covariatet bare skal være et estimat af intensiteten
# Det giver os så en model fittet på den log lineære form som vi gerne vil have.
# Valget af sigma specificerer vores bånbredde.
# Vi kan bruge year2 som kovariat til year1, for at kunne "se ind i fremtiden"
covariate <- density.ppp(year2, sigma = 0.05)
model <- kppm(year1~covariate, clusters = "LGCP")
real <- simulate(model, drop = TRUE)
plot(real)
points(year1, col = 'red')

cov21 <- density.ppp(year2, sigma = 0.05)
cov10 <- density.ppp(year1, sigma = 0.05)
model <- kppm(year0~cov21+cov10, clusters = "LGCP")
real <- simulate(model, drop = TRUE)
plot(real)
points(year0, col = 'red')

# Vi kan teste om vores fit til dataen er ekstrem
num_sim <- 100
alpha_est <- 0.05
rank <- ceiling(alpha_est*num_sim/2)
alpha <- 2*rank/num_sim
e <- envelope(model, Lest, nsim=num_sim, nrank=rank)
plot(e, .-r~r)

# vi kører samme show for shot noise process
cov21 <- density.ppp(year2, sigma = 0.05)
cov10 <- density.ppp(year1, sigma = 0.05)
model <- kppm(year0~cov21+cov10, clusters = "Thomas", statistic = "pcf")
real <- simulate(model, drop = TRUE)
plot(real)
points(year0, col = 'red')

# Vi kan teste om vores fit til dataen er ekstrem
num_sim <- 100
alpha_est <- 0.05
rank <- ceiling(alpha_est*num_sim/2)
alpha <- 2*rank/num_sim
e <- envelope(model, Lest, nsim=num_sim, nrank=rank)
plot(e, .-r~r)

# Det ser ud til at vores shot noise process ikke passer lige så godt som vores
# log gaussiske process, ud fra vore envelope.

