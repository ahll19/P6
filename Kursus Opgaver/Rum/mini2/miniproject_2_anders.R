library("spatstat")
data <- lapply(murchison,rescale,s=1000,unitname="km")
gold <- data$gold
faults <- data$faults
greenstone <- data$greenstone

# plot dataen fo at få en ide om hvordan det ser ud
plot(murchison)

# Her plotter vi greenstone sammen med gold for at se om der er noget guld som findes væk fra greenstone
# Det kan vi se at der er, et specifikt tilfælde er øverst til højre
plot(greenstone)
points(gold)

# Her plotter vi faults sammen med gold for at se om der er nogen betydelig sammenhæng der.
# Det ser ud til at der er en stor sammenhæng imelle guldets placering og de faults der er beskrevet
plot(faults)
points(gold)

# Vi kan bruge J funktionen (defineret i lektion 5 slide 12) til at bestemme om vores data
# opfører sig som en stationær poisson process, er mere regulær, eller "klumper" mere sammen.
# Vi ville forvente at se at vores data er mere "klumpet", hvilket ville være repræsenteret med
# J(r) < 1 for r i (0, R) for et fornuftigt R.
# Hvad vi ser er også rigtig meget hvad vi forventede, da værdien for J hurtigt falder under 1
J <- Jest(gold)
plot(J)

# Da der er så stor en sammenklumpning, som befinder sig på faults det meste af tide,
# kan vi vælge at lave en model hvor intensiteten for guld-processen har eksponentielt henfald som
# vi rejser væk fra et fault
fault_distance <- distmap.psp(faults)
plot(fault_distance)
points(gold)
# Ved at prøve en masse værdier af i et for loop og hand-tune sigma ser 25 ud il at være en god værdi
# for båndbredden af kernen
fault_kernel <- density.psp(faults, 25)
plot(fault_kernel, main="40")
points(gold)

model <- ppm(data$gold~fault_distance, Poisson(), method="mpl")
# giver ingen coeficienter....