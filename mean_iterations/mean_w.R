# Bayes_Mean
# Group Mean estimate with heteroscesdastic uncertainties
# INPUT:
# x - measurement
# e_x - measurement uncertainty
# 
# Return mean,  mean standard error, and quantiles
# Robust to outliers 
coin_unpack <- function(x){as.data.frame(as.matrix(x))}
require(R2jags)
Bayes_Mean <- function(x,e_x){
  nobs = 5
  model.data <- list(x = x,                
                     e_x = e_x,              
                     N = nobs)    

jags_model <-"
model{
# Likelihood
for (i in 1:N){
 x[i] ~ dnorm(mu,pow(e_x[i],-2))
}
# Weakly uniform prior for mean
    mu ~ dnorm(-1,1e-1) # mean
}"

params <- c("mu")
# Run mcmc
mufit <- jags(data =  model.data,
                 parameters = params,
                 model.file = textConnection(jags_model),
                 n.chains = 3,
                 n.thin = 10,
                 n.iter = 5000,
                 n.burnin = 1000)
quant <- quantile(as.numeric(mufit$BUGSoutput$sims.list$mu),probs=c(0.16,0.5,0.84))
#mux <- median(as.numeric(mufit$BUGSoutput$sims.list$mu))
e_mux <- mad(as.numeric(mufit$BUGSoutput$sims.list$mu))
mux <- quant[2]
e_low <- quant[1]
e_up <- quant[3]
return(data.frame(mean = mux, e_low = e_low, e_up = e_up, sd = e_mux))
}


#djoint <-  djoint0 %>% filter(case=="72SNIa28SNII")
#Bayes_Mean(djoint$stan_w_lowz,djoint$stan_wsig_lowz)

djoint0 <- data.table::fread("joint_summary_cases_omprior.csv")

djoint <- djoint0 %>% group_by(case) %>%
  summarise(Bayes = Bayes_Mean(stan_w_lowz,stan_wsig_lowz))  %>% 
  coin_unpack() 
djoint  

write.csv(djoint,"summary_w_post.csv",row.names=FALSE)

