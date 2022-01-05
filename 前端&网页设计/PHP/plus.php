 <?php 

  echo "<br>";
  $n1=$_POST['n1'];
  $n2=$_POST['n2'];
  $n3=$_POST['n3'];
  $f1=(float)$n1;
  $f2=(float)$n2;
  $f3=(float)$n3;

  echo "the result of ".$f1."+".$f2."=".$f3."  is:  ";
if($f3 == $f1+$f2)
   {
    echo "right" ;}
else
    {  
    echo "false <br>" ;
    }



?>

