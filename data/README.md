The `Text.zip` file comes from [ICPSR study 06148](https://www.icpsr.umich.edu/web/NACJD/studies/6148), *Observing and Interviewing Offenders in St. Louis, 1989-1990*, by Richard Wright and Scott Decker.

It is created with this code:

    import pandas as pd
    
    # This is from ICPSR 06148, Decker/Wright interviewss
    file = open("06148-0001-Data.txt","r")
    content = file.read()
    file.close()
    
    # split on totally blank line
    res = content.split("                                                                               ")
    
    # get rid of whitespace
    fin = []
    for l in res:
        if l == "\n":
            pass
        else:
            l2 = l.split("\n")
            l2 = [r.strip() for r in l2 if len(r.strip()) > 0]
            l2 = ' '.join(l2)
            l2 = l2.replace(" â- Month and Day removedã","")
            l2 = l2.replace(":",": ")
            l2 = l2.replace("  "," ").replace("  "," ").strip()
            fin.append(l2)
    
    
    res_df = pd.DataFrame(fin,columns=['Text'])
    res_df.to_csv('Text.zip',index=False)

I save the zipped up file to github just because it is less of a space hog.