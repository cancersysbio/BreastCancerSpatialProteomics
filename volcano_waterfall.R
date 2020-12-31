library(dplyr)
library(lmerTest)
library(tidyr)
library(reshape)
library(ggplot2)
library(ggrepel)
library(ggeffects)
library(vioplot)

# read in data as "data.norm" this is the data frame that is used in the functions below

tumor.probe.names = c("Beta.Catenin","pS6","PTEN","P.ERK","S6","Ki.67", "AKT","PSTAT3","p.AKT","Her2",
                      "Pan.Cytokeratin")
immune.probe.names = c("CD8","CD4","CD68","CD14","GZMB","CD3","CD66B","VISTA","PD1","PD.L1","IDO.1",
                       "CD44","CD56","CD45","CD19","CD45RO","STING.TMEM173","Lag3", "CD11c","ICOS.CD278",
                       "CD27","CD163","OX40L.CD252.TXGP1","B7.H3","B7.H4.VTCN1", "FOXP3", "X4.1BB")
general.probe.names = c("Beta.2.microglobulin","Bcl.2")
all.probes = c(tumor.probe.names, immune.probe.names, general.probe.names)

# example code used to generate waterfall plots
waterfall.plot <- function(pcr, point, roi_type, er.status,ylab) {
  df_dsp <- gather(data.norm, key='Gene',value='count', 9:(length(colnames(data.norm))))
                   # the 9 is the first column of protein data, should be modified accordingly
  genes=unique(df_dsp$Gene)
  df_dsp_m <- melt(df_dsp, id.vars = c('dsp_roi_uid','patient','timepoint','panck','no.pCR','ER', 'Gene'), 
                   measure.vars = 'count')
  df_t_DE <- subset(df_dsp_m,  timepoint %in% point & panck %in% roi_type & no.pCR %in% pcr & ER %in% er.status)
  df_t_DE$no.pCR <- factor(df_t_DE$no.pCR, levels = pcr)
  df_t_DE$panck <- factor(df_t_DE$panck, levels = roi_type)
  df_t_DE$patient <- factor(df_t_DE$patient)
  
  p_tumor <- data.frame(gene = genes)
  p_tumor$FC = p_tumor$p_value <- NA
  
  for(gene in genes) {
    tmp <- df_t_DE[df_t_DE$Gene == gene, ]
    mod <- lmer(formula = 'value ~ timepoint +(1|patient)', data = tmp)
    # if instead of looking at longitudinal changes you want to want too look at changes between panCK+ and panCK- for
    # instance, this lmer formula should be changed
    cf <- coefficients(summary(mod))
    p_tumor[p_tumor$gene == gene, 'FC'] <- cf[2,1]
    p_tumor[p_tumor$gene == gene, 'p_value'] <- cf[2,5]
  }
  p_tumor$Significance <- -log10(p_tumor$p_value)
  p_tumor$FDR <- p.adjust(p_tumor$p_value, 'fdr')
  p_tumor$Significance <- -log10(p_tumor$FDR)
  p_tumor <- p_tumor[!is.na(p_tumor$FC), ]
  class = vector(length=nrow(p_tumor))
  class[which(p_tumor$gene %in% tumor.probe.names)]="tumor"
  class[which(p_tumor$gene %in% immune.probe.names)]="immune"
  class[which(p_tumor$gene %in% general.probe.names)]="general"
  p_tumor = cbind(p_tumor,class)
  
  o <- order(p_tumor$FC,decreasing=TRUE,na.last=NA)
  p_tumor_o <- p_tumor[o,]
  color.for.waterfall = vector(length=length(p_tumor_o$class))
  color.for.waterfall[which(p_tumor_o$class=="tumor")]=alpha("#8DC6EF",.45)
  color.for.waterfall[which(p_tumor_o$class=="immune")]=alpha("#FCD5B5",.45)
  border.for.waterfall = vector(length=length(p_tumor_o$class))
  border.for.waterfall[which(p_tumor_o$FDR<.05)]="black"
  border.for.waterfall[which(p_tumor_o$FDR>=.05)]="white"
  bp <- barplot(p_tumor_o$FC, col=color.for.waterfall, border = border.for.waterfall, ylim = c(-3,2),
                ylab = ylab)
  text((bp+.25), .2, p_tumor_o$gene,cex=1,pos=3, srt=90)
  legend("bottomleft", c("General","Tumor", "Immune", "Signficant Fold Change"),box.lty=0,
         fill = c(alpha("#FCD5B5",.45),alpha("#8DC6EF",.45),alpha("#FCD5B5",.45), "white"),
         border = c("white", "white", "white", "black"))
  return(p_tumor)
}
# example call of the waterfall function
pcr <- c(0,1) # 1 = no pCR
point <- c('B','R')
roi_type <- c(1)
er.status <- c(1)
filename = 'output.pdf'
xlab = expression('Change On-treatment/Pre-treatment, log'[2]*'(PanCK-E)')
#pdf(filename, width=12,height = 6)
er_pos= waterfall.plot(pcr, point, roi_type, er.status,xlab)
#dev.off()

# example code for generating volcano plots (e.g. figures 2a and 2b)
volcano.plot <- function(pcr, point, roi_type, er.status,xlab) {
  df_dsp <- gather(data.norm, key='Gene',value='count', 25:(length(colnames(data.norm))-1))
  genes=unique(df_dsp$Gene)
  df_dsp_m <- melt(df_dsp, id.vars = c('dsp_roi_uid','patient','timepoint','panck','no.pCR','ER', 'Gene'), 
                   measure.vars = 'count')
  df_t_DE <- subset(df_dsp_m,  timepoint %in% point & panck %in% roi_type & no.pCR %in% pcr & ER %in% er.status)
  df_t_DE$no.pCR <- factor(df_t_DE$no.pCR, levels = pcr)
  df_t_DE$panck <- factor(df_t_DE$panck, levels = roi_type)
  df_t_DE$patient <- factor(df_t_DE$patient)
  
  p_tumor <- data.frame(gene = genes)
  p_tumor$FC = p_tumor$p_value <- NA
  
  for(gene in genes) {
    tmp <- df_t_DE[df_t_DE$Gene == gene, ]
    mod <- lmer(formula = 'value ~ timepoint+ (1|patient)', data = tmp)
    # if instead of looking at longitudinal changes you want to want too look at changes between panCK+ and panCK- for
    # instance, this lmer formula should be changed
    cf <- coefficients(summary(mod))
    p_tumor[p_tumor$gene == gene, 'FC'] <- cf[2,1]
    p_tumor[p_tumor$gene == gene, 'p_value'] <- cf[2,5]
  }
  p_tumor$Significance <- -log10(p_tumor$p_value)
  p_tumor$FDR <- p.adjust(p_tumor$p_value, 'fdr')
  p_tumor$Significance <- -log10(p_tumor$FDR)
  p_tumor <- p_tumor[!is.na(p_tumor$FC), ]
  class = vector(length=nrow(p_tumor))
  class[which(p_tumor$gene %in% tumor.probe.names)]="tumor"
  class[which(p_tumor$gene %in% immune.probe.names)]="immune"
  class[which(p_tumor$gene %in% general.probe.names)]="general"
  p_tumor = cbind(p_tumor,class)

  ggplot(p_tumor, aes(x = FC, y = Significance, size = Significance, label = gene, color=class)) +
    ylim(0,8) + xlim(-3,2)+ geom_point() +
    scale_color_manual(values=c(alpha("#FCD5B5",.45),alpha("#8DC6EF",.45)))+
    geom_hline(yintercept = -log10(0.05), linetype="longdash", col="red") +
    geom_hline(yintercept = 0) +
    geom_vline(xintercept = 0) +
    geom_text_repel(data = subset(p_tumor, FDR<0.05), size = 4.5, show.legend = F,
                    point.padding = .3, box.padding = .3, min.segment.length = .2, color="black") +
    theme_light(base_size = 15) +
    labs(x = xlab,
         y = expression('Significance, -log'[10]*' (FDR adjusted P-value)')) +
    guides(color = guide_legend(override.aes = list(size = 5)))
}

# example to run volcano plot code
pcr <- c(1) # 0 = pCR
point <- c('B','R')
roi_type <- c(1)
er.status <- c(0,1)
filename = 'volc_PanCK-E_B_R_protein_npcr.pdf'
xlab = expression('Change Runin/Baseline, log'[2]*'(PanCK-E, no pCR)')
pdf(filename)
volcano.plot(pcr, point, roi_type, er.status,xlab)
dev.off()