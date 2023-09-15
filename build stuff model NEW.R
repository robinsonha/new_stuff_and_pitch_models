
library(tidyverse)
options(scipen=999)
library(xgboost)
library(modelr)
library(tidymodels)
library(mlr)

# read in data, bind, remove excess dfs
s2017 <- read_csv('statcast_2017.csv')
s2018 <- read_csv('statcast_2018.csv')
s2019 <- read_csv('statcast_2019.csv') %>% filter(game_date > "2019-03-27")
s2021 <- read_csv('statcast_2021.csv')
s2022 <- read_csv('statcast_2022.csv') %>% filter(game_date > "2022-04-06")
s2023 <- read_csv('statcast_2023.csv')

### Four total stuff models are built, one for each p_throws / stand combination. For each one, filters must be changed below 

mlbraw <- bind_rows(s2017, s2018, s2019, s2021, s2022, s2023) %>% distinct() %>% filter(p_throws == "L", stand == "L", game_type == "R")
rm(s2022, s2021, s2019, s2018, s2017, s2023)


### Filter out pitchers hitting besides Ohtani
pitchers <-  mlbraw %>% group_by(pitcher) %>% summarize(pitches=n()) %>% ungroup() %>% filter(pitches > 74, pitcher != "660271")

## remove NA rows for needed variables, flip values for lefties, create year column
mlbraw1 <- mlbraw %>% anti_join(pitchers, by = c("batter"="pitcher")) %>%
  filter(description != "pitchout", balls < 4, strikes < 3, outs_when_up < 3, !is.na(release_speed), !is.na(release_spin_rate), !is.na(release_extension),
         !is.na(release_pos_x), !is.na(release_pos_z), !is.na(p_throws), !is.na(stand), !is.na(zone), !is.na(plate_x), !is.na(plate_z),
         !pitch_type %in% c("EP", "PO", "KN", "FO", "CS", "SC", "FA"), !is.na(spin_axis), !is.na(pfx_x), !is.na(pfx_z), !is.na(delta_run_exp),
         !str_detect(des, "pickoff"),!str_detect(des, "caught_stealing"), !str_detect(des, "stolen_"), !des %in% c("game_advisory", "catcher_interf")) %>%
  mutate(release_pos_x = ifelse(p_throws == "R", release_pos_x, -release_pos_x),
         pfx_x = ifelse(p_throws == "R", pfx_x, -pfx_x), out_of_zone = if_else(zone > 9, 1, 0),
         spin_axis = ifelse(p_throws == "R", spin_axis, -spin_axis), year = year(as.Date(game_date)),
         events = case_when(
           events %in% c("double_play", "triple_play", "field_error", "field_out", "fielders_choice", "fielders_choice_out",
                         "force_out", "grounded_into_double_play") ~ "out",
           events %in% c("sac_fly", "sac_bunt", "sac_fly_double_play", "sac_bunt_double_play") ~ "sacrifice",
           TRUE ~ events),
         pitch_type = case_when(
           pitch_type %in% c("FT", "SI") ~ "SI",
           pitch_type %in% c("CU", "KC") ~ "CU",
           pitch_type %in% c("ST", "SL", "SV") ~ "SL",
           TRUE ~ pitch_type), count = paste(balls, strikes, sep="-"), 
         matchup= paste(p_throws, stand, sep ="HPvs")) %>%
  mutate(above_zone = plate_z-sz_top, below_zone = sz_bot-plate_z, zone_x_max = plate_x-.83, zone_x_min = plate_x+.83
  )

## Find average delta run values for each ball or strike
bs_vals <- mlbraw1 %>% filter(type != "X") %>%  group_by(type) %>% summarize(dre_bs = mean(delta_run_exp, na.rm=T))

## now balls in play
ip_filt <- mlbraw1 %>% filter(type == 'X')

event_lm <- lm(delta_run_exp ~ events, data=ip_filt)
summary(event_lm)

## derive dre values for balls in play 
ip_vals <- ip_filt %>% add_predictions(event_lm, var = "pred_dre") %>%  group_by(events) %>% 
  summarize(dre_ip = mean(pred_dre, na.rm=T))

## merge values into main dataframe
mlbraw1 <- mlbraw1 %>% left_join(bs_vals, by = "type") %>%
  left_join(ip_vals, by = "events") %>% 
  mutate(dre_final = case_when(
    type != "X" ~ dre_bs,
    TRUE ~ dre_ip))



## establish each pitcher's primary fastball by season, then join to df
pitcher_fastballs <- mlbraw1 %>% filter(pitch_type %in% c("FF", "FC", "SI")) %>% group_by(pitcher, year) %>% 
  summarize(fb_velo = mean(release_speed), fb_max_ivb = quantile(pfx_z, .8), fb_max_x = quantile(pfx_x, .8), fb_min_x = quantile(pfx_x, .2), 
            fb_max_velo = quantile(release_speed, .8),fb_axis = mean(spin_axis, na.rm=T))

mlbraw2 <- mlbraw1 %>% left_join(pitcher_fastballs, by = c("year", "pitcher")) %>% mutate(spin_dif = spin_axis - fb_axis, velo_dif = release_speed-fb_velo,
                           ivb_dif = fb_max_ivb-pfx_z, break_dif = (fb_max_x*.5+fb_min_x*.5)-pfx_x)

##"set" is a marker used further down to manually control variable subsetting (can be omitted if this is to be automated)
if(mlbraw2$p_throws[1] == 'L' & mlbraw2$stand[1] == 'R' & mlbraw2$game_type[1] == "R"){
  mlbraw2$set<-1
} else if(mlbraw2$p_throws[1] == 'L' & mlbraw2$stand[1] == 'L' & mlbraw2$game_type[1] == "R"){
  mlbraw2$set<-2
} else if(mlbraw2$p_throws[1] == 'R' & mlbraw2$stand[1] == 'L' & mlbraw2$game_type[1] == "R"){
  mlbraw2$set<-3
} else {
  mlbraw2$set<-4
}
  
# take 2
final_vars <- mlbraw2 %>% select(dre_final, starts_with("fb_"), release_speed, release_spin_rate,
                                 release_extension, release_pos_x, release_pos_z, pfx_x, pfx_z,
                                 pitch_type, spin_axis, spin_dif, velo_dif, ivb_dif, break_dif, set) 


## create dummy variables where needed
library(caret)
dmy <- dummyVars(" ~ pitch_type", data = final_vars)
trsf <- data.frame(predict(dmy, newdata = final_vars))

### join data
vars <- cbind(final_vars, trsf) %>% select(-pitch_type) %>% filter(!is.na(dre_final))


# Remove all objects except "vars" dataframe
rm(list = setdiff(ls(), "vars"))
gc()

#start modeling
set.seed(4813)

##Test for correlations between variables
print(summary(vars))
corr_vars <- rcorr(as.matrix(vars))
flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  =(cormat)[ut],
    p = pmat[ut]
  )
}
mat<-flattenCorrMatrix(corr_vars$r, corr_vars$P)
view(mat)

##Create an alternative data table with one of each pair of highly correlated (>0.95) variables removed
##Supervise and complete this manually
vars<-vars %>%
  select(-c('fb_velo','spin_dif',))

##OR TO AUTOMATE THIS PROCESS:
#mat<-data.frame(mat)
#mat<-mat[mat$r>0.95,]
#vars<-vars[,!names(vars) %in% mat$column]


#split into training (80%) and testing set (20%)
# vars <- vars %>% sample_n(300000)
c_pitching_split <- initial_split(vars, strata = dre_final, prop = .7) 
train <- training(c_pitching_split)
test <- testing(c_pitching_split)

## try hyperpram testing
traintask <- makeRegrTask (data = train,target = "dre_final")
testtask <- makeRegrTask (data = test,target = "dre_final")

rm(train, test)

#create learner
lrn <- makeLearner("regr.xgboost")
lrn$par.vals <- list( objective="reg:squarederror", eval_metric="rmse")

#set parameter space
params <- makeParamSet( makeDiscreteParam("booster",values = "gbtree"),
                        makeDiscreteParam("tree_method",values = "hist"),
                        makeIntegerParam("max_depth",lower = 3L,upper = 14L), 
                        makeIntegerParam("min_child_weight",lower = 3L,upper = 10L), 
                        makeNumericParam("subsample",lower = 0.3,upper = 1), 
                        makeNumericParam("colsample_bytree",lower = 0.3,upper = 1),
                        makeIntegerParam("gamma",lower = 0L,upper = 5L),
                        makeIntegerParam("alpha",lower = 0L,upper = 5L),
                        makeDiscreteParam("nrounds", 
                                          values = c(50, 100, 200, 300, 400, 500, 600, 700)),
                        makeDiscreteParam("eta",
                                          values = c(.01, .03, .05, .075, .1, .15, .2)))

#set resampling strategy
rdesc <- makeResampleDesc("CV", iters=5L)

#search strategy
ctrl <- makeTuneControlRandom(maxit = 25L)
gc()
#set parallel backend
library(parallel)
library(parallelMap) 
parallelStartSocket(cpus = detectCores())

# parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc,
                     par.set = params, control = ctrl, show.info = T)

sqrt(mytune$y)


#set hyperparameters
lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)

#train model
xgmodel <- train(learner = lrn_tune,task = traintask)
imp <- mlr::getFeatureImportance(xgmodel)

view(imp$res)

#predict model 
xgpred <- predict(xgmodel,testtask) %>% as.data.frame()

caret::RMSE(xgpred$truth, xgpred$response)


## retrain model on full dataset and save for future use
fulltask <- makeRegrTask (data = vars,target = "dre_final")

xgmodel <- train(learner = lrn_tune,task = fulltask)

saveRDS(xgmodel, file = "c_stuff_LHPvsLHBNEW.rds")
