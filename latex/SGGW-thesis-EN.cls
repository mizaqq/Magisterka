%
% Class for writing diploma theses - batchelor, engineer and master theses
% in English at the Warsaw University of Life Sciences - SGGW
%

\newcommand{\SGGWthesisVersion}{1.071b2}
% For each version in Polish a corresponding version in English is issued.

% Version 1.071b2 of 2024.11.12
% Author: Leszek Chmielewski
% Changes:
% 1. The images have officially accepted English contents now.
%
% Version 1.071b1 of 2024.11.05
% Author: Leszek Chmielewski
% Changes:
% 1. Small change: the variable with the version no. moved to the beginning of document.
% 2. Updated comment in \ProvidesClass{}[]
% 3. Small changes in source main-EN.tex which update the changes for newer class version.

% Wersja 1.07 z 2024.10.22
% Autorzy: Leszek Chmielewski, Dawid Puszka
% Zmiany:
% 1. Poprawka polegająca na dodaniu pakietu "lmodern" w takm celu, aby
%    w dokumencie PDF były użyte naturalne polskie czcionki, zamiast
%    dostawiania polskich znaków diakrytycznych do liter łacińskich,
%    co powodowało, że widok dokumentu był prawidłowy, lecz kopiowanie
%    tekstu powodowało rozdzielanie polskich liter i znaków diakrytycznych.
% 2. Bibliografia jest umieszczana w spisie treści jako pozycja nienumerowana.
%    Zarządzenie Rektora tego nie normuje, ale jest to przyjęty zwyczaj.
% 3. Uporządkowanie kolejności wczytywania pakietów.
% 4. Dodanie w przykładowej pracy "main.tex", dostarczanej wraz z klasą,
%    notatek informujących o zgodności klasy z Zarządzeniami Rektora SGGW.
% 5. Obecnie zarządzenie Rektora i wytyczne podane są na stronie:
%    https://www.sggw.edu.pl/studenci/praca-dyplomowa/

% Wersja 1.062 z 2023.07.31
% Autorzy: Konrad Jażdżyk, Patryk Stępień
% Zmiana:
% 1. Poprawka wielkości marginesów.

% Wersja 1.061 z 2022.01.26
% Autor: Leszek Chmielewski
% Zmiany:
% 1. Drobny retusz na stronie oświadczeń promotora i autora: "Świadom" -> "Świadom/a".
% 2. W rzeczywistości nowe zarządzenie Rektora i wytyczne podane są na stronie:
%    https://www.sggw.edu.pl/strona-glowna/studia/informacje-praktyczne-dla-studentow/praca-dyplomowa/

% Wersja 1.06 z 2021.11.14
% Autorzy: Mikołaj Tkaczyk, Mikołaj Piekutowski
% Zmiany:
% 1. Zmiana strony tytułowej, aby zgadzała się z Zarządzeniem Rektora SGGW nr 100 z dnia 22 września 2021 r.
% http://student.wzim.sggw.pl/wp-content/uploads/2015/10/Zarz%C4%85dzenie-Nr-100-Rektora-SGGW-z-dnia-22-wrze%C5%9Bnia-2021-r.-wytyczne-dotycz%C4%85ce-przygotowywania-prac-dyplomowych.pdf
% 2. Możliwość ustawienia nagłówka strony tytułowej za pomocą ustawienia flag INZYNIERSKA, LICENCJACKA lub MAGISTERSKA.
% 3. Dodanie warunkowego ustawiania stopki strony tytułowej, kontrolowanej na razie przez jedną flagę WZIM.
% 4. Zmiana wersji na zmienną, aby dynamicznie wyświetlała się w notce strony tytułowej.
% 5. Zmiany te wymagały dodania dodatkowych plików obrazów, które muszą być umieszczone w tym samym katalogu, co plik klasy.

% Wersja 1.05 z 2019.06.25
% Autorzy: Pawel Wasil, Bartosz Sowa
% Zmiany:
% 1. Zmiana \onehalfspacing na \setstretch{1.4} - dzieki temu interlinia jest niemal identyczna jak w wersji w Word, czyli odtep 1.5.
%
% Wersja 1.04 z 2018.01.17.
% Autorzy: Pawel Wasil, Michal Rosa
% Wspolpraca: Leszek Chmielewski
% Zmiany:
% 1. Poprawiono blad wielkosci czcionki tytulu pracy.
% 2. Drobne retusze w formatach

% Wersja 1.03 z 2018.01.10.
% Autor: Łukasz Adamczyk
% Zmiany:
% 1. Poprawiono blad zwiazany z odstepami linii tekstu tytulu polskiego i angielskiego.

% Wersja 1.02 z 2018.01.08.
% Autor: Leszek Chmielewski <leszek_chmielewski@sggw.pl>
% Wspolpraca: Michal Rosa, Pawel Wasil
% Zmiany:
% 1. Poprawka w wielkosciach czcionek na pierwszych stronach.
% 2. Poprawka w wielkosci interlinii - teraz jest 1.5 poza stronami: tytulowa i oswiadczen.
% 3. Globalne zmniejszenie odstepow w listach - enumerate, itemize, description.
%    Zastosowano pakiet "enumitem". Jesli jest taka potrzeba, mozna korzystac
%    z jego opcji globalnie (w preambule) lub w konkretnych listach.
%    Moze to byc potrzebne w listach wielopoziomowych.
% 4. Koncowe oswiadczenie bedzie na stronie przedostatniej, zawsze nieparzystej.
%    Ta strona i otaczajace ja puste strony nie beda numerowane.

% Wersja 1.01 z 2017.12.13.
% Autor: Leszek Chmielewski <leszek_chmielewski@sggw.pl>
% Zmiany:
% 1. Poprawka stylu podpisow rys i tab. Zastosowano pakiet "caption".

% Wersja 1.0 z 2017.01.16.
% Autor: Łukasz Adamczyk
%
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{SGGW-thesis-EN}[2024/11/04 LaTeX class for writing theses in English at WULS-SGGW]


\DeclareOption{multip}{\@multiptrue}
\DeclareOption{onep}{\@multipfalse}

\newif\if@multip
\ExecuteOptions{multip}
\@multipfalse

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\DeclareOption*{}
\ProcessOptions*\relax
\LoadClass[12pt,notitlepage,twoside]{report}

\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{mathptmx} %font Times New Roman
\RequirePackage{lmodern} %for Polish font in UTF8 in PDF files

\RequirePackage[nottoc]{tocbibind} %bibliography in Contents without chapter number

\RequirePackage[labelfont=bf,labelsep=period]{caption} %titles of figures and tables: bold font, full stop

\RequirePackage[a4paper,top=2.5cm,bottom=2.5cm,inner=3.5cm,outer=2.5cm]{geometry}

\RequirePackage{babel} % for multi-language support

\RequirePackage{graphicx}

\RequirePackage{titlesec} % format change for chapters to (1 Name)
\titleformat{\chapter}
  {\normalfont\LARGE\bfseries}{\thechapter}{1em}{}
\titlespacing*{\chapter}{0pt}{3.5ex plus 1ex minus .2ex}{2.3ex plus .2ex}

\RequirePackage{enumitem} %change of distances in lists - general reduction
% If the effect is not suitable, then options of the package "enumitem" should be used in relevant lists.
\setlist[itemize]{itemsep=2pt,parsep=0pt,partopsep=0pt,topsep=2pt}
\setlist[description]{itemsep=2pt,parsep=0pt,partopsep=0pt,topsep=2pt}
\setlist[enumerate]{itemsep=2pt,parsep=0pt,partopsep=0pt,topsep=2pt}

\RequirePackage{setspace} %change of verse distance - \onehalfpsacing in the text and \singlespacing in statements
\singlespacing

\RequirePackage{xifthen}
% provides for that the chapter follwing this command will start in an odd page
\newcommand{\startchapterfromoddpage}{\ifthenelse{\isodd{\value{page}}}{\newpage\thispagestyle{empty}\null}{}}

\renewcommand{\normalsize}{\fontsize{12}{13}\selectfont}

\newcommand{\Overnormal}[1]{\@setfontsize\Overnormal{14pt}{16}{#1}}	%font definition 14pt
\newcommand{\VeryLarge}[1]{\@setfontsize\VeryLarge{16pt}{18}{#1}}	%font definition 16pt
\newcommand{\TitleFont}[1]{\@setfontsize\TitleFont{18pt}{22}{#1}}	%font definition 18pt
\newcommand{\VeryVeryLarge}[1]{\@setfontsize\VeryVeryLarge{24pt}{28}{#1}} %font definition 24pt

%definitions of variables
\def\@album{}
\newcommand{\album}[1]{
  \def\@album{#1}
}

\def\@Ptitle{}
\newcommand{\Ptitle}[1]{
  \def\@Ptitle{#1}
}
\def\@thesis{}
\newcommand{\thesis}[1]{
  \def\@thesis{#1}
}
\def\@course{}
\newcommand{\course}[1]{
  \def\@course{#1}
}
\def\@promotor{}
\newcommand{\promotor}[1]{
  \def\@promotor{#1}
}
\def\@pworkplace{}
\newcommand{\pworkplace}[1]{
  \def\@pworkplace{#1}
}

\newif\ifENGINEER
\newif\ifBACHELOR
\newif\ifMASTER
\newcommand{\conditionalthesistypeheader}{
  \ifENGINEER
    \includegraphics[width=9.81cm]{header_eng.png}\par
  \else
    \ifBACHELOR
      \includegraphics[width=9.81cm]{header_bsc.png}\par
    \fi
    \ifMASTER
      \includegraphics[width=9.81cm]{header_msc.png}\par
    \fi
  \fi
}

\newif\ifWZIM
\newcommand{\conditionaldepartmentfooter}{
  \ifWZIM
    \includegraphics[width=11.04cm]{footer_WZIM.png}\par
  \fi
}

\renewcommand{\maketitle}{ %definition of title page
	\begin{flushleft}
		\thispagestyle{empty}
		\vspace*{3.25cm}
		\conditionalthesistypeheader
		\vspace{0.85cm}
		{\normalsize{\@author}}\\
		{\normalsize{\@album}}\\
		\vspace{16px}
        {\TitleFont{\bfseries{\@title}}\par}
		\vspace{0.2cm}
		{\Overnormal{\@Ptitle}\par}
		\vspace{22.5px}
		{\normalsize{\@thesis}}\\
		\vspace{0.1cm}
		{\normalsize{\@course}}\\
    \end{flushleft}
	\begin{flushright}
        \vspace{57px}
		{\normalsize{Thesis under the supervision of}}\\
		{\normalsize{\@promotor}}\\
		{\normalsize{\@pworkplace}}\\
		\vspace{1cm}
	\end{flushright}
	\begin{flushleft}
	    {\normalsize{Warsaw, year \@date}}\\
	    \vfill
	    \conditionaldepartmentfooter
	\end{flushleft}
	
	
	\newpage
	\if@multip
	\else
		
		\null
		\pagestyle{empty}
		\newpage
		\pagestyle{plain}
	\fi
}
\newcommand{\twoppage}[6]{ %definition of page with two authors of the thesis
	\null
	\vfill
	\noindent{\normalsize{The thesis was prepared by a team of:}}\\
	\begin{quote}
	\noindent{\Overnormal{1. #1}}\\
	{\normalsize{#2\\
	The author of:\\
	#3\\
	}}
	\end{quote}
	\vfill
	\begin{quote}
	\noindent{\Overnormal{2. #4}}\\
	{\normalsize{#5\\
	The author of:\\
	#6\\
	}}
	\end{quote}
	\vfill
	\thispagestyle{empty}
	\newpage
	\pagestyle{plain}
}

\newcommand{\threeppage}[9]{ %definition of page with three authors of the thesis
	\null
	\vfill
	\noindent{\normalsize{The thesis was prepared by a~team of:}}\\
	\begin{quote}
		\noindent{\Overnormal{1. #1}}\\
		{\normalsize{#2\\
		The author of:\\
		#3\\
		}}
	\end{quote}
	\vfill
	\begin{quote}
		\noindent{\Overnormal{2. #4}}\\
		{\normalsize{#5\\
		The author of:\\
		#6\\
		}}
	\end{quote}
	\vfill
	\begin{quote}
		\noindent{\Overnormal{3. #7}}\\
		{\normalsize{#8\\
		The author of:\\
		#9\\
		}}
	\end{quote}	
	\vfill
	\thispagestyle{empty}
	\newpage
	\pagestyle{plain}

}
\newcommand{\statementpage}{ %definicja strony z oswiadczeniami
	\null
	\vfill
	\begin{center}
		{\textbf{Declaration of the Thesis Supervisor}}\\
	\end{center}
		I hereby declare that
		\if@multip
			the chapters of the thesis prepared as a team indicated by the author have been prepared under my supervision, and I~confirm that they meet
		\else
			this thesis has been prepared under my supervision, and I~confirm that it meets
		\fi
		the conditions for presenting this work in the procedure for the award of a professional title.\\
				
		\noindent
		Date \dotfill
		\hspace{2cm}
		Supervisor's signature \dotfill

	\vfill
	\begin{center}
		{\textbf{Declaration of the Author of the Thesis}}\\
	\end{center}
		Aware of legal liability, including criminal liability for submitting a false declaration, I hereby declare that
		\if@multip
			the chapters of the diploma thesis prepared as a team were written by myself and did not contain
		\else
			this diploma thesis was written by myself and did not contain
		\fi
		the content obtained in a manner inconsistent with applicable law, in particular the Act of 4 February 1994 on copyright and related rights (Journal of Laws of 2019, item 1231, as amended).\\[2ex]
        I declare that the submitted work has not previously been the basis for any procedure related to awarding a diploma or obtaining a professional title.\\
		
		\noindent
		I certify that this work version is identical to the attached electronic version.\\

		\noindent
		I acknowledge that the diploma thesis will be subject to the procedure of the anti-plagiarism.\\[2ex]
		
		\noindent
		Date \dotfill
		\hspace{2cm}
		Author's signature \dotfill
	\vfill
	\pagestyle{empty}	
	\newpage
	\null
	\pagestyle{empty}
	\newpage
	\pagestyle{plain}
}
\newcommand{\abstractpage}[6]{ %definicja strony ze streszczeniami

    \setstretch{1.4} % od tego miejsca interlinia 1.5; pierwotnie ustawiono \onehalfspacing ale obecna wersja bardziej odpowiada wersji .doc

	\null
	\vfill
	\begin{center}
		\textbf{Streszczenie}\\
	\end{center}
		\noindent
		\textbf{#1}\\[1.5ex]
		\foreignlanguage{polish}{{#2}}
		\\[4ex]
		Słowa kluczowe -- \foreignlanguage{polish}{{#3}}
	
	\vfill
	\newpage
	\begin{center}
		\textbf{Summary}\\
	\end{center}
		\noindent
		\textbf{#4}\\[1.5ex]
		{#5}
		\\[4ex]
		Keywords -- {#6}
	\vfill
	\pagestyle{empty}
	\newpage
	\null
	\pagestyle{empty}
	\newpage
	\pagestyle{plain}
}
\newcommand{\beforelastpage}[1][]{ %definicja strony przedostatniej; zawsze nieparzysta
	\newpage\thispagestyle{empty}\null
    \ifthenelse{\isodd{\value{page}}}{}{\newpage\thispagestyle{empty}\null}

    \singlespacing % z powrotem interlinia pojedyncza
	\thispagestyle{empty}
	\null
	\vspace{5\baselineskip}
	\begin{flushleft}
		Wyrażam zgod\k{e} na udost\k{e}pnienie mojej pracy w~czytelniach Biblioteki SGGW w~tym w~Archiwum Prac Dyplomowych SGGW%
		\ifthenelse{\isempty{#1}}{}{po roku #1}.
	\end{flushleft}
	\vspace{2cm}
	\begin{flushright}
		\hfill\hfill\dotfill\\
		(czytelny podpis autora pracy)
	\end{flushright}
	\vfill
	\newpage\null
    \thispagestyle{empty}
}
\sloppy
\clubpenalty4000
\widowpenalty4000
\sfcode`\.=1000\relax

% Paragraph spacing and indentation:
\parskip=0pt plus 1pt
\parindent=15pt
\raggedbottom

% Float placement parameters:
\renewcommand{\topfraction}{0.99}       % Max fraction of a page for floats at top
\renewcommand{\textfraction}{0.01}      % Min fraction of page that must be text
\renewcommand{\dbltopfraction}{0.7}     % For double-column floats, max top float fraction
\setlength{\floatsep}{13pt plus 2pt minus 2pt}      % Space between floats
\setlength{\textfloatsep}{20pt plus 2pt minus 2pt}  % Space between floats and text
\setlength{\intextsep}{13pt plus 2pt minus 2pt}     % Space around in-text floats
\setlength{\dbltextfloatsep}{20pt plus 2pt minus 4pt} % Space around double-column floats
\setlength{\dblfloatsep}{13pt}                       % Space between double-column floats
