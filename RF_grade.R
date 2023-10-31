
# RF ML model for Grade 02/23/23 -------------------------------------
library(tidymodels)
library(tidyverse)
library(here)


# data load ---------------------------------------------------------------
grd_lvl <- c("LTB", "B", "X42", "X46", "X52", "X60", "X65", "X70")

chem <-
  # read_csv(here('all_data_2023.csv')) %>%
  read_csv("C:\\Users\\Joel\\OneDrive - RSI Pipeline Solutions\\ChemistryGrade\\all_data.csv") %>%
  filter(wall > 0) %>%
  select(c,
         mn,
         s,
         #including Sulfur only increased accuracy 0.1% when using ys
         OD,
         wall,
         grade,
         ys,
         uts
         ) %>%
         mutate(
           DT = OD / wall,
           grade = ifelse(grade == "Early", "LTB", grade),
           grade = ifelse(grade == "Grade B", "B", grade),
           grade = factor(grade)
         ) %>%
           filter(
             grade == "X42" & ys >= 41 |
               grade == "X52" & ys >= 51 |
               grade == "X60" & ys >= 59 |
               grade == "B" & ys >= 34 |
               grade == "X46" & ys >= 45 |
               grade == "X65" & ys >= 64 |
               grade == "X70"  & ys >= 69 |
               grade == "LTB",

             ys > 0,
             s < 0.1,
             c > 0.02
           ) %>%
           # select(-ys) %>%
           drop_na() #%>%
           # unique()


# Train/Test Split --------------------------------------------------------

trainSplit <- initial_split(chem, strata=grade, prop = 0.85)
trainData <- training(trainSplit)
testData <-  testing(trainSplit)

# Do Parallel -------------------------------------------------------------

all_cores <- parallel::detectCores(logical = FALSE)
doParallel::registerDoParallel(all_cores)

# Recipe ------------------------------------------------------------------

grade_recipe <-
  recipes::recipe(grade ~ . , data = trainData) %>%
  # convert categorical variables to factors
  recipes::step_string2factor(all_nominal()) %>%
  # # combine low frequency factor levels
  # recipes::step_other(all_nominal(), threshold = 0.01) %>%
  # # remove no variance predictors which provide no predictive information
  prep()

train_processed <- bake(grade_recipe,  new_data = training(trainSplit))


# Model Spec --------------------------------------------------------------

RF_model <-
  parsnip::rand_forest(
    mode = "classification",
    trees = 500,
    min_n = 1,
    mtry = 3 # use mtry =2 for comp only and mtry = 3 for YS & UTS
  ) %>%
  set_engine("ranger",
             importance = "impurity",
             num.threads = 5) %>%
  translate()

RF_model

# Workflow ----------------------------------------------------------------
# We use the new tidymodels workflows package to add a formula to our XGBoost
# model specification.

RF_wf <-
  workflows::workflow() %>%
  add_model(RF_model) %>%
  add_formula(grade ~ .)

# Fit Model and Collect Metrics ---------------------------------------------

## apply fit() to workflow
model_fit <- fit(RF_wf, data = train_processed)

model_fit$fit

RF_aug <- augment(model_fit, new_data = testData)

metrics(RF_aug, truth =grade,estimate =.pred_class)

forest <-
  randomForest::randomForest(
    grade ~ .,
    data = trainData,
    localImp = TRUE,
    mtry = 3,
    ntree = 500,
    importance = TRUE
  )

# Confusion Matrix --------------------------------------------------------

cm <- RF_aug %>%
  mutate(
    .pred_class = factor(.pred_class, levels = grd_lvl),
    grade = factor(grade, levels = grd_lvl)
  ) %>%
  conf_mat(truth = grade, estimate = .pred_class)

cm

# Variable Importance -----------------------------------------------------

forest %>%
  vip::vip(aesthetics = list(fill = 'steelblue')) +
  theme_bw()


# New Data ---------------------------------------------------------

## Color Spec for plots
grd_lvl <- c("LTB", "B", "X42", "X46", "X52", "X60", "X65", "X70")

pal <-
  c(
    "#e6194B",
    "#f58231",
    "#ffe119",
    "#3cb44b",
    "#42d4f4",
    "#4363d8",
    "#911eb4",
    "#f032e6"
  )

`%!in%` <- Negate(`%in%`)

path <- "C:\\Users\\Joel\\OneDrive - RSI Pipeline Solutions\\ChemistryGrade\\stations\\Syar Industries\\data"


new_dat <- read_csv (glue::glue('{path}\\Syar Ind Lake Herm_inputs_All_NWT1-OD1_2023-10-20.csv')
 ) %>%
  janitor::clean_names() %>%
  rename(wall = pipe_wt,
         OD =pipe_od,
         ys = yield_strength,
         uts = tensile_strength) %>%
  mutate(DT = OD/wall,
         ys = ys/1000,
         uts = uts/1000) %>%
  drop_na()



# new_dat <- valid

# New Predictions ---------------------------------------------------------

# fit_new_dat <-  bind_cols(new_dat,predict(model_fit, new_dat, type="prob")) %>%
#   mutate(model="Chem only",
#          # scenario = "c = 1.0"
#          ) %>%
#   rename(.pred_LTB = .pred_Early,
#          .pred_B = '.pred_Grade B')



# fit_new_dat <- augment(model_fit, new_data = new_dat )
stamp <- format(Sys.time(), '%m_%d_%Y_%H%M')


fit_new_dat <- augment(model_fit, new_data = new_dat) %>%
  mutate(timestamp = stamp)

write_csv(fit_new_dat,glue::glue('{path}\\predictions_comp_ys_uts_{stamp}.csv'))

vars <- tibble(vars =names(chem)) %>%
  mutate(timestamp = stamp)

write_csv(vars, glue::glue('{path}\\variables_comp_ys_uts_{stamp}.csv'))


# Write Predictions -------------------------------------------------------

# write_csv(fit_new_dat,  "C:\\Users\\Joel\\OneDrive - RSI Pipeline Solutions\\ChemistryGrade\\stations\\Goleta\\Goleta_grade_preds_chem_only.csv")

# Mean Probability by Grade ------------------------------------------------
# features <- c("JT-01", "JT-04","JT-07")
# fit_nmz <- tibble(nm = names(fit_new_dat)) %>%
#   filter(nm!= '.pred_class', str_detect(nm,'.pred')) %>%
#   mutate(nm = str_remove(nm, '.pred_'))
#
#
# other <- tibble(
#   feature = fit_new_dat$feature[1],
#   name = grd_lvl[grd_lvl %!in% fit_nmz$nm],
#   chemistry = "Filings",
#   sum_tot = 0
# )


fit_new_dat %>%
  # rename(chemistry = composition_method) %>%
select(-.pred_class) %>%
  pivot_longer(cols = starts_with('.pred')) %>%
  mutate(name = str_remove(string = name,
                           pattern = ".pred_"),
         name = factor(name, levels = grd_lvl)
  ) %>%
  mutate(name = factor(name, levels = grd_lvl)) %>%
  group_by(joint, composition_method , name) %>%
  summarise(sum_tot = mean(value)) %>%
  ggplot(aes(name, sum_tot)) +
  geom_col(aes(fill = name),
           position = "dodge",
           show.legend = F) +
  geom_text(aes(label = round(sum_tot,2)), vjust = -0.5,size =4) +
  scale_y_continuous(breaks = scales::pretty_breaks(), limits = c(0,1))+
  facet_wrap( joint ~ composition_method  ,
              ncol=3) +
  theme_bw(12) +
  scale_fill_manual(values = pal) +
  labs(
    title = "Syar Industries Station Mean Probabilities by Grade",
    # subtitle = "Using Different YS",+
    y = "Mean Probability",
    x = "Grade",
    caption = paste0("Composition +YS + UTS model\n",format(Sys.time(), '%m/%d/%Y %H:%M'))
  )+
  theme(plot.margin = margin(0.7,0.7,0.7,0.7,"cm"),
        panel.spacing.x = unit(0.7,"cm") )


# ridgeline plot ----------------------------------------------------------

fit_new_dat %>%
  filter(chemistry == "Filings") %>%
  pivot_longer(.pred_LTB:.pred_X70) %>%
  mutate(
    name = str_remove(string = name,
                      pattern = ".pred_"),
    name = factor(name, levels = grd_lvl)
  ) %>%
  filter(name %in% c("LTB", "B", "X42", "X52")) %>%
  ggplot(aes(value)) +
  ggridges::geom_density_ridges(aes(y = scenario, fill = scenario),
                                show.legend = FALSE,
                                alpha = 0.5) +
  facet_wrap(feature ~ name)+
  theme_bw()

# Resampling --------------------------------------------------------------

grade_folds <- vfold_cv(chem,
                        strata = grade,
                        v = 10,
                        repeats = 45)

rf_wf <-
  workflow() %>%
  add_model(RF_model) %>%
  add_formula(grade ~ .)

rf_fit_cv <-
  rf_wf %>%
  fit_resamples(grade_folds)


perf <- collect_metrics(rf_fit_cv)

mets <- rf_fit_cv %>%
  unnest(.metrics) %>%
  filter(.metric == "accuracy") %>%
  select(.estimate) %>%
  summarise(
    mean = mean(.estimate),
    Std = sd(.estimate),
    uci = mean + 1.96 * Std,
    lci = mean - 1.96 * Std
  );mets

rf_fit_cv %>%
  unnest(.metrics) %>%
  filter(.metric == "accuracy") %>%
  ggplot(aes(.estimate)) +
  geom_histogram(aes(y = after_stat(density)),
                 fill = 'slateblue2',
                 col = 'black') +
  labs(x = "Accuracy",
       y = NULL) +
  stat_function(
    fun = dnorm,
    args = list(mets$mean, mets$Std),
    col = 'red',
    linewidth = 1.1
  ) +
  annotate("text", x= 0.86, y = 40, label = glue::glue('Mean = {round(mets$mean*100,1)}%\n95 CI = {round(mets$lci*100,1)} to {round(mets$uci*100,1)}%'), size =5)+
  # annotate("text", x= 0.86, y = 37, label = glue::glue('95 CI = {round(mets$lci*100,1)} to {round(mets$uci*100,1)}%'), size =5)+
  scale_x_continuous(labels = scales::label_percent())+
  theme_bw(14)+
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        plot.margin = ggplot2::margin(0.5,0.5,0.5,0.5,"cm")
  )+
  labs(title = "Grade Prediction Model Accuracy",
       subtitle = "10 Fold Cross-Validation Repeated 45 Times",
       caption = "With S & without D/t included")
