Version: see bottom for latest version numner and a change log list

A few notes about this:

(1) 
The entire document is configured using configuration.tex.
Just fill out all these commands specified there.
Additionally, adapt mainfile.tex according to your purposes. 

(2)
Compilation: execute the makefile. Read Section 4.2 "Building the PDF"
of the pre-compiled for further details. (See next point!)

(3)
The subfolder "pre-compiled-PDFs" contains pre-compiled PDFs.
One for each available title sheet style (so you can choose the one
you like the most), but more importantly one document, thus showing
all the advice I want to convey.

(3.a)
Note that you can choose the layout of your preference (in the
above-mentioned configuration file).

(3.b)
Please, read the PDF (see above) completely!
It contains valuable advice on how to write a project report.
I advice to come back to this PDF even at later stages of the
project (i.e., after you wrote some parts already) as odds are
high you did not follow all the advice since it might not have
been relevant to you yet.


Pascal
pascal.bercher@anu.edu.au


version history:

// note that some details mention section numbers. They can however change of course. Sorry...


1.092 24.4.2024
          - Fixed a typo

1.091 16.4.2024
          - Fixed a few typos and missing words.

1.09 18.3.2024
          - Expanded the correctness part of bibtex entries; I made a subchecklist more explicit
            but also added an entry that's done wrong super often (capitalization)

1.081 5.2.2024
          - Fixed a bug on title page: Number of pts are now ignored when it is an Honours thesis
            // Note that only the 'main PDF' got updated; the four example title PDF still need to be updated
1.08   24.1.2024
           - added a new section on using quotes correctly (it should always be 66/99!)
           - updated the huge block of comments at the top of the main file. It was terribly outdated.
           - one section title had a linebreak in it; I removed it just for the TOC (it looks better now)
           - added another disclaimer to the marking section (in the very beginning)
           - reverted this history to have the newest change at the top
           - moved the folder with pre-compiled PDFs from the source code folder to the root
1.071  5.12.2023
           - added an entry about publication date of citations to the marking criteria
1.07    26.11.2023
           - added section on marking (assessment of the work)
1.06    26.11.2023
            - Section 5.1, which was titles, is now on capitalization rules (including titles) and
              thus contains new content (on how to cite figures, tables, etc.). The content about
              number of subsections and their introductory text was moved out.
            - Section 5.2 on general rules to increase appearance now contains the advice
              that was moved out of Section 5.1 (see above). 
            - Section 6.1 on dots in LaTeX, now also mentions dots in parentheses.
            - Section 6.8 on Algorithms was added.
1.05     19.11.2023
             - re-compiled all cover PDF pages (e.g., college names were not up to date,
               and some spacing wasn't right). Also added a fifth PDF with the entire report,
               so that we have one PDF per cover *and* the report separately.
             - optimized configuration file: 
               * points don't have to be defined for Honours theses (24 is now set automatically)
               * Command for second supervisor got renamed into "twoOrMore..." as it also
                  allows to define an arbitrary number of additional supervisors
              - Acknowledgement explanations got expanded. It also contained a comment about 
                the title page which was now moved to 4.1, where it fits better. 
              - Section 4 and 4.1: The entire first page got simplified compressing it to its core.
              - Section 4.2 on building the PDF got simplified.
              - Major restructuring: Previously Section 4.3, containing all my advice on thesis
                writing is now separated into two own chapters: one for general advice and one
                for advice specifically on LaTeX.
              - Evaluation: Added subsections to expand advice significantly.    
1.04     11.4.2023
             - fixed "By" (textbf) on cover page.
1.03     16.12.2022
            - updated college name and fixed some spelling errors (thanks, chatGPT!)
1.02     27.10.2022
            - added third and fourth supervisor to config file (if required)
1.01     29.9.2022
            - added two more pieces of advice to the main section:
              * on how to include definitions and theorems and how to write them and
              * some starting points for how to write mathemathical equations/formulae
1.001   25.9.2022
            - only this readme was improved and some files were moved.
1.00     early 2022 (approximately)
            - very first version! All is new! 

            
            
// ToDo list for future versions:

Show how program code can be included easily.
--> Jinghang Feng (PhD student from school of biology)
