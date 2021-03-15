library(mlogit)

wd <- "/path/to/application/"
setwd(wd)

long_df <- read.csv('data/long_data_rhone.csv')
long_df$choice <- as.logical(long_df$choice)
long_df$has_car <- as.logical(long_df$has_car)

long_df$CS1 <- relevel(factor(unlist(long_df$CS1)), 'Employee')
long_df$DIPL <- relevel(factor(unlist(long_df$DIPL)), 'Secondary')

data <- dfidx(long_df, idx = c("obs_id", "alt_id"), choice = "choice")

quadratic <- FALSE
logarithmic <- FALSE

if (quadratic) {
  nl <- mlogit(choice ~ 1 + car_per_indiv + has_car + tt_cycling + tt_cycling_2
               + tt_car + tt_car_2 + tt_motorcycle + tt_motorcycle_2
               + tt_public_transit + tt_public_transit_2 + tt_walking
               + tt_walking_2 + tt_x_AGEREVQ + tt_x_SEXE + tt_x_artisan
               + tt_x_bluecollar + tt_x_executive + tt_x_farmer
               + tt_x_intermediate | AGEREVQ + CS1 + SEXE,
               data = data,
               weights = unlist(long_df["IPONDI"], use.names=FALSE),
               reflevel = "car",
               #heterosc = TRUE,
               #nests = list(motorized = c("car", "motorcycle"), soft = c("cycling", "walking"), public = c("public transit")),
               #un.nest.el = TRUE
  )  
} else if (logarithmic) {
  nl <- mlogit(choice ~ 1 + car_per_indiv + has_car + tt_cycling
               + log_tt_cycling + tt_car + log_tt_car + tt_motorcycle
               + log_tt_motorcycle + tt_public_transit + log_tt_public_transit
               + tt_walking + log_tt_walking + tt_x_AGEREVQ + tt_x_SEXE
               + tt_x_artisan + tt_x_bluecollar + tt_x_executive + tt_x_farmer
               + tt_x_intermediate | AGEREVQ + CS1 + SEXE,
               data = data,
               weights = unlist(long_df["IPONDI"], use.names=FALSE),
               reflevel = "car",
               #heterosc = TRUE,
               #nests = list(motorized = c("car", "motorcycle"), soft = c("cycling", "walking"), public = c("public transit")),
               #un.nest.el = TRUE
  )
} else {
  nl <- mlogit(choice ~ 1 + car_per_indiv + has_car + tt_cycling + tt_car
               + tt_motorcycle + tt_public_transit
               + tt_walking 
               #+ tt_cycling_2 + tt_walking_2
               + tt_x_AGEREVQ
               + tt_x_SEXE + tt_x_artisan + tt_x_bluecollar + tt_x_executive
               + tt_x_farmer + tt_x_intermediate | AGEREVQ + CS1 + SEXE,
               data = data,
               weights = unlist(long_df["IPONDI"], use.names=FALSE),
               reflevel = "car",
               #heterosc = TRUE,
               #nests = list(motorized = c("car", "motorcycle"), soft = c("cycling", "walking"), public = c("public transit")),
               #un.nest.el = TRUE
  )
}
print(summary(nl))
write.csv(nl$coefficients, "output/regression_results.csv")

# Compute average value of time.
ids = long_df$choice&long_df$alt_id=="car"
m = 0
#m = m + nl$coefficients["tt_cycling"] * sum(long_df[long_df$alt_id=="cycling", "choice"])
m = m + nl$coefficients["tt_car"] * sum(long_df[long_df$alt_id=="car", "choice"])
#m = m + nl$coefficients["tt_motorcycle"] * sum(long_df[long_df$alt_id=="motorcycle", "choice"])
#m = m + nl$coefficients["tt_public_transit"] * sum(long_df[long_df$alt_id=="public transit", "choice"])
#m = m + nl$coefficients["tt_walking"] * sum(long_df[long_df$alt_id=="walking", "choice"])
m = m + nl$coefficients["tt_x_AGEREVQ"] * sum(long_df[ids, "AGEREVQ"])
m = m + nl$coefficients["tt_x_SEXE"] * sum(long_df[ids, "SEXE"])
m = m + (
  nl$coefficients["tt_x_artisan"]
  * sum(long_df[ids, "CS1"]=="Artisan")
)
m = m + (
  nl$coefficients["tt_x_bluecollar"]
  * sum(long_df[ids, "CS1"]=="Blue-collar")
)
m = m + (
  nl$coefficients["tt_x_executive"]
  * sum(long_df[ids, "CS1"]=="Executive")
)
m = m + (
  nl$coefficients["tt_x_farmer"]
  * sum(long_df[ids, "CS1"]=="Farmer")
)
m = m + (
  nl$coefficients["tt_x_intermediate"]
  * sum(long_df[ids, "CS1"]=="Intermediate")
)
m = - m / sum(ids)
print("Average value of time: ")
print(as.double(m))

write.csv(nl$linpred, 'data/predictions.csv')
