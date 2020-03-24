DaFy is Python software package short for Data Analysis Factory.

It is designed to process data in a pipe-line pattern, which comprises of several 'Data_Filter-->Data_Engine' components.
Filter and Engine packages are stored in seperated Python packages, which can be called in the main script. In principle,
all functions during data analysis can be wrapped into either a filter-like or engine-like objects. To make the main script
concise and readable, we should be forced to do that if you want to make any script contribution to DaFy package.

Inside DaFy, you can write many main scripts for different projects to expand its functionality. To develop a project, you 
write first the associated filter and engine objects to be called in your main script.
