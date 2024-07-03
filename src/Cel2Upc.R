#!/usr/bin/Rscript
suppressPackageStartupMessages({
    library(optparse)
    library(affy)
})

option_list <- list(
    make_option(c("--input"), type="character", default=NULL, help="Directory of CELs"),
    make_option(c("--output"), type="character", default=NULL, help="Output file")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

print(paste("Input Directory:", opt$input))
print(paste("Output file:", opt$output))

cel_data <- ReadAffy(celfile.path = opt$input)
upc_data <- rma(cel_data)
exp_df <- exprs(upc_data)
colnames(exp_df) = gsub(pattern = '.cel.gz', replacement = '', colnames(exp_df), ignore.case = T)

write.csv(exp_df, opt$output, quote = F)

print('DONE.')
