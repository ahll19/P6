library(spatstat)

# opg 2
data = swedishpines # redwood, cells
K = Kest(data)
L = Lest(data)
G = Gest(data)
F = Fest(data)
J = Jest(data)

# a
plot(K)
plot(L)
plot(G)
plot(F)
plot(J)

# b
plot(L, .-r~r)

# c
e <- envelope(data, Lest, nsim=39)
plot(e, .-r~r)

# d
alpha <- 2/(40)
# Hvis vi forkaster ved en eller flere forsøg kan vi ikke bruge samme signifikans-niveau
# eftersom at vi ville forkaste på et tidspunkt hvor vi ellers ikke ville med bare en
# r-værdi. Måske det ender med at indeholde nogle eksponenter så

# e
# nej tak brormand

# f
e <- envelope(data, Lest, nsim=199,nrank=5)
alpha2 <- 2*5/199
plot(e, .-r~r)
# vi har det samme signifikans niveau. Vi får et større bånd ved et større antal
# simuleringer. Vi får også et meget stort signifikans niveua ved et lille antal
# simuleringer ift. rank
# flere simuleringer gør det mere stabilt --> vi skal bruge højere rank for
# at have konstant signifikans niveau
