library("spatstat")
plot(chorley)
split_chorley <- split.ppp(chorley)
lung <- split_chorley$lung
larynx <- split_chorley$larynx
chorley.extra$plotit()
inc <- chorley.extra$incin

# check om cases opfører sig som en poisson process
plot(Kest(chorley))
plot(Hest(chorley))
plot(Lest(chorley), .-r~r)
# Her kan vi se at det ikke passer på en homogen poisson process, hvilket
# man også kunne gætte sig frem til ud fra at se på plottet.
# vi ville ikke regne med at se punkterne være homogen fordelt
# eftersom at vores tilfælde af kræft er afhængig af hvor folk bor (som ikke er homogent)

# Her laver vi nogle envolopes, bare lige for sjov
num_sim <- 9
alpha_est <- 0.05
rank <- ceiling(alpha_est*num_sim/2)
alpha <- 2*rank/num_sim
e <- envelope(chorley, Lest, nsim=num_sim, nrank=rank)
plot(e, .-r~r)
# og selvfølgelig passer det ikke ind

# Her estimerer vi et population map ved at lave et heat map over lung cancer
for (i in 1:6){
  plot(density.ppp(lung, i*0.2))
}
lambda_pop <- density.ppp(lung, 1.2)
# Vi vælger en sigma på 1.2 fordi det virker fornuftigt nok

# plot afstand til incenirator fra punkterne
lambda_dist <- distmap(as.ppp(inc, chorley))
plot(lambda_dist)

# Vi prøver at fit vores første model til dataen, og laver en realisering
model <- ppm(split(chorley)$larynx~lambda_pop+lambda_dist,Poisson(),method="mpl")
beta0 <- model$coef[1]; beta1 = model$coef[2]; beta2 = model$coef[3];
model_run <- exp(beta0 + beta1*lambda_pop + beta2*lambda_dist)

# cool plots
plot(model_run) # estimat af hvor det er mere sandsynligt at få larynx cancer
points(larynx)  # tilfælde af hvor man vil få larynx cancer
points(chorley.extra$incin, col = 'red', pch = 10, cex = 4)

# mangler lige noget mere analyse af dataen
# kig på z test for at tjekke signifikans af parameter