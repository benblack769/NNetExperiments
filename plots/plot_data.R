library(tidyverse)
library(readr)


args <- commandArgs(TRUE)

directory = args[1]
filename = args[2]
use_every = strtoi(args[3])
use_first_vecs = strtoi(args[4])

in_file_name = paste(directory,filename,sep="/")
out_file_name = paste(directory,"_plots","/",filename,"png",sep="")

filt_data = read_tsv(in_file_name) %>%
  filter(TIME_STAMP%%use_every==1,
         SOURCE_ID < use_first_vecs) %>%
plot = ggplot(filt_data,aes(x=TIME_STAMP,y=VALUE,color=SOURCE_ID)) + 
    geom_line()

ggsave(out_file_name,plot)