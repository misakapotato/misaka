<div style="width:1200px;background-color: lightpink;background-size:100% 100%;height: 300px;margin: 0 auto;">
	<div style="width: 40%;float: left;height:300px;font-family: 幼圆;text-align:left;font-size:14px;font-family:微软雅黑 light,微软雅黑,'Courier New', Courier, monospace;margin: 10px auto;font-weight: bold;color:grey;text-align:center;">
		<script>
			var num;//定义num变量
			num = Math.random()*100;
			num = Math.floor(num);
			document.write("加法验算器这次刷新的幸运值："+num+"<br>");
		</script>
		<form action="plus.php" method="post">
	
			输入一个数字:<input type="text" name="n1" size="15" maxlength="25" ><br>
			再输一个数字:<input type="text" name="n2" size="15" maxlength="25" ><br>
			你计算的结果:<input type="text" name="n3" size="15" maxlength="25" ><br><br>
			<input type="submit" value="验算"  name="submit" >
			<input type="reset" value="归零"  name="reset" >
		</form>
	</div>
	<div style="width: 20%;float: left;height:300px;"><p style="text-align:center">二次元加减验算器</p><img src="web01.png" style="height:240px;">
	</div>
    <div style="width: 40%;float: right;height:300px;font-family: 幼圆;text-align:left;font-size:14px;font-family:微软雅黑 light,微软雅黑,'Courier New', Courier, monospace;margin: 10px auto;font-weight: bold;color:grey;text-align:center;">
		<script>
			var num;//定义num变量
			num = Math.random()*100;
			num = Math.floor(num);
			document.write("减法验算器这次刷新的幸运值："+num+"<br>");
		</script>
		<form action="minus.php" method="post">
			输入一个数字:<input type="text" name="n1" size="15" maxlength="25" ><br>
			再输一个数字:<input type="text" name="n2" size="15" maxlength="25" ><br>
			你计算的结果:<input type="text" name="n3" size="15" maxlength="25" ><br><br>
		<input type="submit" value="验算"  name="submit" >
		<input type="reset" value="归零"  name="reset" >
		</form>
	</div>
</div>


